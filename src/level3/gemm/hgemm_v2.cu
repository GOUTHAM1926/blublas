#include "mycublas.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>

// HGEMM V2: 128x128x64 Tile, 4 Warps, 3-Stage Pipeline + CTA Swizzle
// Based on Hgemm Strided Batched V24/BK64 & BGEMM V5
// =====================================================================

#define BM 128
#define BN 128
#define BK 64
#define STAGES 3
#define THREADS_PER_BLOCK 128
#define SWIZZLE 8

#define MMA_M16N8K16_F32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1) \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 " \
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};" \
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) \
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

__global__ void __launch_bounds__(128, 1)
hgemm_v2_kernel(
    int M, int N, int K,
    __half alpha,
    const __half* __restrict__ A, int lda,
    const __half* __restrict__ B, int ldb,
    __half beta,
    __half* __restrict__ C_ptr, int ldc,
    int grid_x, int grid_y)
{
    // --- Robust CTA Swizzling (1-to-1 Mapping) ---
    int bx = blockIdx.x;
    int by = blockIdx.y;
    const int swizzle_factor = 8;
    if (gridDim.y % swizzle_factor == 0) {
        const int block_idx = blockIdx.y * gridDim.x + blockIdx.x;
        by = (block_idx % swizzle_factor) + (block_idx / (gridDim.x * swizzle_factor)) * swizzle_factor;
        bx = (block_idx / swizzle_factor) % gridDim.x;
    }

    if (by * BM >= M || bx * BN >= N) return;

    extern __shared__ __half s_mem[];
    const int s_step_a    = BM * BK;        
    const int s_step_b    = BK * BN;        
    const int s_stage_size = s_step_a + s_step_b;

    const int tid  = threadIdx.x;
    const int wid  = tid >> 5;
    const int lane = tid & 31;
    const int wy   = wid / 2;
    const int wx   = wid % 2;

    float acc[4][8][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 8; j++)
            for (int l = 0; l < 4; l++) acc[i][j][l] = 0.0f;

    auto load_to_stage = [&](int stage, int k_off) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int r = tid;
            int c = i * 8;
            const __half* g_ptr = A + (long long)(by * BM + r) * lda + (k_off + c);
            int swizzled_c = c ^ ((r & 7) << 3);
            uint32_t sm_addr = __cvta_generic_to_shared(&s_mem[stage * s_stage_size + r * BK + swizzled_c]);
            int pred = (by * BM + r < M && k_off + c < K) ? 16 : 0;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm_addr), "l"(g_ptr), "r"(pred));
        }
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int r = tid / 2;
            int c = (tid % 2) * 64 + i * 8;  
            const __half* g_ptr = B + (long long)(k_off + r) * ldb + (bx * BN + c);
            int swizzled_c = c ^ ((r & 7) << 3);
            uint32_t sm_addr = __cvta_generic_to_shared(&s_mem[stage * s_stage_size + s_step_a + r * BN + swizzled_c]);
            int pred = (k_off + r < K && bx * BN + c < N) ? 16 : 0;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm_addr), "l"(g_ptr), "r"(pred));
        }
    };

    load_to_stage(0, 0); asm volatile("cp.async.commit_group;\n");
    if (BK < K) load_to_stage(1, BK); asm volatile("cp.async.commit_group;\n");

    int write_stage = 2, read_stage  = 0;
    uint32_t frA[2][4][4], frB[2][8][2];

    asm volatile("cp.async.wait_group 1;\n");
    __syncthreads();

    auto load_frA = [&](uint32_t reg[4], int ki, int mi, int st) {
        int r = wy * 64 + mi * 16 + ((lane / 8) % 2) * 8 + (lane % 8);
        int c = ki + (lane / 16) * 8;
        int swizzled_c = c ^ ((r & 7) << 3);
        uint32_t addr = __cvta_generic_to_shared(&s_mem[st * s_stage_size + r * BK + swizzled_c]);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                     : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3]) : "r"(addr));
    };

    auto load_frB = [&](uint32_t reg[2], int ki, int ni, int st) {
        int r = ki + ((lane / 8) % 2) * 8 + (lane % 8);
        int c = wx * 64 + ni * 8;
        int swizzled_c = c ^ ((r & 7) << 3);
        uint32_t addr = __cvta_generic_to_shared(&s_mem[st * s_stage_size + s_step_a + r * BN + swizzled_c]);
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
                     : "=r"(reg[0]), "=r"(reg[1]) : "r"(addr));
    };

    #pragma unroll
    for (int i = 0; i < 4; i++) load_frA(frA[0][i], 0, i, read_stage);
    #pragma unroll
    for (int j = 0; j < 8; j++) load_frB(frB[0][j], 0, j, read_stage);

    for (int k = 0; k < K; k += BK) {
        int next_k = k + 2 * BK;
        if (next_k < K) load_to_stage(write_stage, next_k);
        asm volatile("cp.async.commit_group;\n");

        #pragma unroll
        for (int i = 0; i < 4; i++) load_frA(frA[1][i], 16, i, read_stage);
        #pragma unroll
        for (int j = 0; j < 8; j++) load_frB(frB[1][j], 16, j, read_stage);
        #pragma unroll
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 8; j++)
                MMA_M16N8K16_F32(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3], frA[0][i][0], frA[0][i][1], frA[0][i][2], frA[0][i][3], frB[0][j][0], frB[0][j][1]);

        #pragma unroll
        for (int i = 0; i < 4; i++) load_frA(frA[0][i], 32, i, read_stage);
        #pragma unroll
        for (int j = 0; j < 8; j++) load_frB(frB[0][j], 32, j, read_stage);
        #pragma unroll
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 8; j++)
                MMA_M16N8K16_F32(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3], frA[1][i][0], frA[1][i][1], frA[1][i][2], frA[1][i][3], frB[1][j][0], frB[1][j][1]);

        #pragma unroll
        for (int i = 0; i < 4; i++) load_frA(frA[1][i], 48, i, read_stage);
        #pragma unroll
        for (int j = 0; j < 8; j++) load_frB(frB[1][j], 48, j, read_stage);
        #pragma unroll
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 8; j++)
                MMA_M16N8K16_F32(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3], frA[0][i][0], frA[0][i][1], frA[0][i][2], frA[0][i][3], frB[0][j][0], frB[0][j][1]);

        if (k + BK < K) {
            asm volatile("cp.async.wait_group 1;\n");
            __syncthreads();
            read_stage  = (read_stage  + 1) % STAGES;
            write_stage = (write_stage + 1) % STAGES;
            #pragma unroll
            for (int i = 0; i < 4; i++) load_frA(frA[0][i], 0, i, read_stage);
            #pragma unroll
            for (int j = 0; j < 8; j++) load_frB(frB[0][j], 0, j, read_stage);
        }
        #pragma unroll
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 8; j++)
                MMA_M16N8K16_F32(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3], frA[1][i][0], frA[1][i][1], frA[1][i][2], frA[1][i][3], frB[1][j][0], frB[1][j][1]);
    }
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    __half* s_out = s_mem;
    const int rt = lane / 4, ct = (lane % 4) * 2;
    float alpha_f = __half2float(alpha);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            // Tile 128x128. Warp tile should be 64x64.
            int r_idx = wy * 64 + i * 16 + rt;
            int c_idx = wx * 64 + j * 8 + ct;
            s_out[r_idx * BN + c_idx]         = __float2half(acc[i][j][0] * alpha_f);
            s_out[r_idx * BN + c_idx + 1]     = __float2half(acc[i][j][1] * alpha_f);
            s_out[(r_idx+8) * BN + c_idx]     = __float2half(acc[i][j][2] * alpha_f);
            s_out[(r_idx+8) * BN + c_idx + 1] = __float2half(acc[i][j][3] * alpha_f);
        }
    }
    __syncthreads();

    const __half2 h2beta = {beta, beta};
    const __half2 h2alpha = {alpha, alpha};

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int r  = (tid >> 4) + i * 8, c  = (tid & 15) << 3, gr = by * BM + r, gc = bx * BN + c;
        if (gr < M && gc < N) {
            int4 vals = *(int4*)&s_out[r * BN + c];
            if (beta != __float2half(0.0f)) {
                int4 old = *(int4*)&C_ptr[(long long)gr * ldc + gc];
                __half2* h2v = (__half2*)&vals;
                __half2* h2o = (__half2*)&old;
                #pragma unroll
                for (int l = 0; l < 4; l++) h2v[l] = __hadd2(h2v[l], __hmul2(h2o[l], h2beta));
            }
            *(int4*)&C_ptr[(long long)gr * ldc + gc] = vals;
        }
    }
}

extern "C" void mycublasHgemm_v2(
    mycublasHandle_t handle, mycublasOperation_t transa, mycublasOperation_t transb,
    int M, int N, int K, const __half alpha,
    const __half *A, int lda, const __half *B, int ldb,
    const __half beta, __half *C, int ldc)
{
    cudaStream_t stream = (handle != nullptr) ? handle->stream : 0;
    static bool smem_set = false;
    size_t smem_size = (size_t)STAGES * (BM * BK + BK * BN) * sizeof(__half);
    if (!smem_set) {
        cudaFuncSetAttribute(hgemm_v2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size);
        smem_set = true;
    }
    int grid_x = (N + BN - 1) / BN, grid_y = (M + BM - 1) / BM;
    dim3 grid(grid_x, grid_y);
    dim3 block(THREADS_PER_BLOCK);
    hgemm_v2_kernel<<<grid, block, smem_size, stream>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, grid_x, grid_y);
}
