#include "mycublas.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include <algorithm>
#include <iostream>

// ============================================================
// Sgemm V18 — Ultimate Edition (Revision 7)
//
// Highlights:
//   1. Global Pointer Induction: Source pointers are 
//      incremented rather than re-calculated.
//   2. SMEM Pointer Induction: Corrected and Hoisted.
//   3. 128 Threads, 3 Stages, SASS-Like issue patterns.
// ============================================================

#define BM      128
#define BN      128
#define BK      16
#define STAGES  3
#define THREADS 128
#define AS_SIZE    (BM * BK)
#define BS_SIZE    (BK * BN)
#define STAGE_SIZE (AS_SIZE + BS_SIZE)

#define MMA_TF32(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1)                \
    asm volatile(                                                   \
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "     \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"       \
        : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)                     \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1))

template <bool IsAligned>
__global__ void __launch_bounds__(THREADS, 1)
sgemm_SM86_ultimate_kernel(
    int M, int N, int K,
    float alpha,
    const float* __restrict__ A, int lda, long long int strideA,
    const float* __restrict__ B, int ldb, long long int strideB,
    float beta,
    float* __restrict__ C, int ldc, long long int strideC,
    int batchCount)
{

    const int batch = (int)blockIdx.z;
    if (batch >= batchCount) return;

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

    const int tid  = (int)threadIdx.x;
    const int lane = tid & 31, wid = tid >> 5;
    const int wy = wid >> 1, wx = wid & 1;

    // Pointer Induction: Initial Global Pointers
    const float* gA_ptr[4];
    const float* gB_ptr[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const int rowA = by * BM + (tid / 4) + i * 32;
        gA_ptr[i] = A + (long long)batch * strideA + (long long)rowA * lda + (tid % 4) * 4;
        const int rowB = i * 4 + (tid / 32);
        gB_ptr[i] = B + (long long)batch * strideB + (long long)rowB * ldb + (bx * BN + (tid % 32) * 4);
    }

    extern __shared__ float smem[];

    float acc[4][8][4];
    #pragma unroll
    for (int i = 0; i < 4; i++) for (int j = 0; j < 8; j++) for (int d = 0; d < 4; d++) acc[i][j][d] = 0.f;

    auto load_to_stage = [&](int stage, int k_curr) {
        float* As = smem + stage * STAGE_SIZE;
        float* Bs = As + AS_SIZE;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int r = tid / 4 + i * 32, c = (tid % 4) * 4, sc = c ^ ((r & 3) << 2);
            uint32_t sm_a = __cvta_generic_to_shared(&As[r * BK + sc]);
            const int gr = by * BM + r, gc = k_curr + c;
            if constexpr (IsAligned) { 
                int src_bytes = (gr < M) ? max(0, min(16, (K - gc) * 4)) : 0; 
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm_a), "l"(gA_ptr[i]), "r"(src_bytes)); 
            } else { 
                float4 val = {0,0,0,0}; 
                if (gr < M) { if (gc < K) val.x = gA_ptr[i][0]; if (gc+1 < K) val.y = gA_ptr[i][1]; if (gc+2 < K) val.z = gA_ptr[i][2]; if (gc+3 < K) val.w = gA_ptr[i][3]; } 
                *(float4*)&As[r * BK + sc] = val; 
            }
            gA_ptr[i] += BK; // Pointer Induction
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int r = i * 4 + (tid / 32), c_base = (tid % 32) * 4, sc = c_base ^ ((r & 7) << 2);
            uint32_t sm_b = __cvta_generic_to_shared(&Bs[r * BN + sc]);
            const int gr = k_curr + r, gc = bx * BN + c_base;
            if (r < BK) {
                if constexpr (IsAligned) {
                    int src_bytes = (gr < K) ? max(0, min(16, (N - gc) * 4)) : 0;
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm_b), "l"(gB_ptr[i]), "r"(src_bytes));
                } else {
                    float4 val = {0,0,0,0}; if (gr < K) { if (gc < N) val.x = gB_ptr[i][0]; if (gc+1 < N) val.y = gB_ptr[i][1]; if (gc+2 < N) val.z = gB_ptr[i][2]; if (gc+3 < N) val.w = gB_ptr[i][3]; }
                    *(float4*)&Bs[r * BN + sc] = val;
                }
            }
            gB_ptr[i] += (long long)BK * ldb; // Pointer Induction
        }
    };

    const int g_sh = lane / 4, t_sh = lane % 4;
    int rbaseA[4], maskA[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) { rbaseA[i] = (wy * 64 + i * 16 + g_sh) * BK; maskA[i] = ((wy * 64 + i * 16 + g_sh) & 3) << 2; }
    int cbaseB[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) cbaseB[j] = wx * 64 + j * 8 + g_sh;

    auto load_frA = [&](uint32_t reg[4], int ks, int mi, int st) {
        float* As = smem + st * STAGE_SIZE;
        auto ga = [&](int row_idx, int c) { return *(uint32_t*)&As[row_idx + (c ^ maskA[mi])]; };
        reg[0] = ga(rbaseA[mi], ks + t_sh); reg[1] = ga(rbaseA[mi] + 8*BK, ks + t_sh);
        reg[2] = ga(rbaseA[mi], ks + t_sh + 4); reg[3] = ga(rbaseA[mi] + 8*BK, ks + t_sh + 4);
    };

    auto load_frB = [&](uint32_t reg[2], int ks, int ni, int st) {
        float* Bs = smem + st * STAGE_SIZE + AS_SIZE;
        const int r0 = ks + t_sh, r1 = r0 + 4;
        auto gb = [&](int r, int c) { return *(uint32_t*)&Bs[r * BN + (c ^ ((r & 7) << 2))]; };
        reg[0] = gb(r0, cbaseB[ni]); reg[1] = gb(r1, cbaseB[ni]);
    };

    load_to_stage(0, 0); asm volatile("cp.async.commit_group;\n");
    if (BK < K) load_to_stage(1, BK); asm volatile("cp.async.commit_group;\n");
    int write_stage = 2, read_stage = 0; uint32_t frA[2][4][4], frB[2][8][2];
    asm volatile("cp.async.wait_group 1;\n"); __syncthreads();
    for (int i = 0; i < 4; i++) load_frA(frA[0][i], 0, i, read_stage);
    for (int j = 0; j < 8; j++) load_frB(frB[0][j], 0, j, read_stage);

    for (int k = 0; k < K; k += BK) {
        if (k + 2 * BK < K) load_to_stage(write_stage, k + 2 * BK); asm volatile("cp.async.commit_group;\n");
        #pragma unroll
        for (int ks = 0; ks < BK; ks += 8) {
            const int p = (ks / 8) & 1, q = p ^ 1;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                if (ks + 8 < BK) { load_frA(frA[q][i], ks + 8, i, read_stage); load_frB(frB[q][i * 2], ks + 8, i * 2, read_stage); load_frB(frB[q][i * 2 + 1], ks + 8, i * 2 + 1, read_stage); }
                else if (k + BK < K) { if (i == 0) { asm volatile("cp.async.wait_group 1;\n"); __syncthreads(); read_stage = (read_stage+1)%STAGES; write_stage = (write_stage+1)%STAGES; }
                    load_frA(frA[q][i], 0, i, read_stage); load_frB(frB[q][i * 2], 0, i * 2, read_stage); load_frB(frB[q][i * 2 + 1], 0, i * 2 + 1, read_stage); }
                #pragma unroll
                for (int j = 0; j < 8; j++) MMA_TF32(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3], frA[p][i][0], frA[p][i][1], frA[p][i][2], frA[p][i][3], frB[p][j][0], frB[p][j][1]);
            }
        }
    }

    const int g_epi = lane / 4, t_epi = lane % 4;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            const int r0 = by * BM + wy * 64 + i * 16 + g_epi, r8 = r0 + 8, col = bx * BN + wx * 64 + j * 8 + t_epi * 2;
            float* C_batch = C + (long long)batch * strideC;
            auto sf2 = [&](int r, int c, float f0, float f1) {
                if (r >= M || c >= N) return;
                float* dst = &C_batch[(long long)r * ldc + c];
                if (c + 1 < N && (((size_t)dst & 7) == 0)) { float2 old = (beta == 0.f) ? make_float2(0,0) : *(float2*)dst; *(float2*)dst = { alpha * f0 + beta * old.x, alpha * f1 + beta * old.y }; }
                else { dst[0] = alpha * f0 + (beta == 0 ? 0 : beta * dst[0]); if (c + 1 < N) dst[1] = alpha * f1 + (beta == 0 ? 0 : beta * dst[1]); }
            };
            sf2(r0, col, acc[i][j][0], acc[i][j][1]); sf2(r8, col, acc[i][j][2], acc[i][j][3]);
        }
    }
}

extern "C" void mycublasSgemmStridedBatched_SM86(
    mycublasHandle_t handle, int M, int N, int K, const float alpha,
    const float* d_A, int lda, long long int strideA,
    const float* d_B, int ldb, long long int strideB,
    const float beta, float* d_C, int ldc, long long int strideC, int batchCount)
{
    int sm_ver, sm_count;
    get_gpu_info(&sm_ver, &sm_count);

    const int gx = (N + BN - 1) / BN;
    const int gy = (M + BM - 1) / BM;

    // Tier 2: 64x64 tile — improved occupancy for medium shapes, scaled by SM count
    // For SM89, avoid 64x64 if we have enough tiles to justify 128x128 (Threshold: 80)
    int tile64_threshold = (sm_ver >= 89) ? 80 : sm_count;
    if (gx * gy < tile64_threshold) {
        mycublasSgemmStridedBatched_nn_tile64_SM86(handle, M, N, K, alpha,
                                                   d_A, lda, strideA, d_B, ldb, strideB,
                                                   beta, d_C, ldc, strideC, batchCount);
        return;
    }

    cudaStream_t stream = (handle != nullptr) ? handle->stream : 0;

    // Tier 3: Delegate to the SplitK variant when the output tile count is small but K is
    // large. For SM89, we use a higher threshold (300) to account for more SMs.
    int sk_threshold = (sm_ver >= 89) ? 300 : 160;
    if (gx * gy < sk_threshold && K >= 1024) {
        mycublasSgemmStridedBatched_splitk_SM86(
            handle, M, N, K, alpha,
            d_A, lda, strideA, d_B, ldb, strideB,
            beta, d_C, ldc, strideC, batchCount);
        return;
    }

    const bool aligned = (((size_t)d_A & 15) == 0) && ((lda & 3) == 0) && (((size_t)d_B & 15) == 0) && ((ldb & 3) == 0);
    static const size_t smem_bytes = STAGES * STAGE_SIZE * sizeof(float);

    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(sgemm_SM86_ultimate_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
        cudaFuncSetAttribute(sgemm_SM86_ultimate_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
        attr_set = true;
    }

    dim3 grid(gx, gy, batchCount);
    if (aligned) sgemm_SM86_ultimate_kernel<true><<<grid, THREADS, smem_bytes, stream>>>(M,N,K,alpha,d_A,lda,strideA,d_B,ldb,strideB,beta,d_C,ldc,strideC,batchCount);
    else sgemm_SM86_ultimate_kernel<false><<<grid, THREADS, smem_bytes, stream>>>(M,N,K,alpha,d_A,lda,strideA,d_B,ldb,strideB,beta,d_C,ldc,strideC,batchCount);
}
