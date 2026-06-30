#include "mycublas.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>

using namespace nvcuda;

// ============================================================
// Hgemm Addmm V34 — Fused Matmul + Bias (FP16)
//
// Highlights:
//   1. Derived from Sgemm_addmm_v34 (Fused Bias + Vectorized Epilogue)
//   2. Uses mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
//   3. Uses ldmatrix for efficient shared memory loads
// ============================================================

#define BM      128
#define BN      128
#define BK      32
#define STAGES  3
#define THREADS 128
#define AS_SIZE (BM * BK)
#define BS_SIZE (BK * BN)
#define STAGE_SIZE (AS_SIZE + BS_SIZE)

#define MMA_M16N8K16_F32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1) \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 " \
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};" \
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) \
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

template <bool IsAligned>
__global__ void __launch_bounds__(THREADS, 2)
hgemm_addmm_v34_kernel(
    int M, int N, int K,
    __half alpha,
    const __half* __restrict__ A, int lda, long long strideA,
    const __half* __restrict__ B, int ldb, long long strideB,
    __half beta,
    const __half* __restrict__ bias, int64_t bias_numel,
    __half* __restrict__ C, int ldc, long long strideC,
    int batchCount)
{
    const int batch = (int)blockIdx.z;
    if (batch >= batchCount) return;

    // --- Robust CTA Swizzling ---
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

    const __half* A_batch = A + (long long)batch * strideA;
    const __half* B_batch = B + (long long)batch * strideB;
    __half*       C_batch = C + (long long)batch * strideC;

    extern __shared__ __half s_mem[];
    
    float acc[4][8][4];
    #pragma unroll
    for (int i = 0; i < 4; i++) for (int j = 0; j < 8; j++) for (int d = 0; d < 4; d++) acc[i][j][d] = 0.f;

    auto load_to_stage = [&](int stage, int k_curr) {
        // Load A (128x32)
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int r = tid; 
            int c = i * 8;
            const __half* g_ptr = A_batch + (long long)(by * BM + r) * lda + (k_curr + c);
            int swizzled_c = c ^ ((r & 3) << 3); 
            uint32_t sm_addr = __cvta_generic_to_shared(&s_mem[stage * STAGE_SIZE + r * BK + swizzled_c]);
            int pred = (by * BM + r < M && k_curr + c < K) ? 16 : 0;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm_addr), "l"(g_ptr), "r"(pred));
        }
        // Load B (32x128)
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int r = tid / 4; 
            int c = (tid % 4) * 8 + i * 32; 
            const __half* g_ptr = B_batch + (long long)(k_curr + r) * ldb + (bx * BN + c);
            int swizzled_c = c ^ ((r & 7) << 3); 
            uint32_t sm_addr = __cvta_generic_to_shared(&s_mem[stage * STAGE_SIZE + AS_SIZE + r * BN + swizzled_c]);
            int pred = (k_curr + r < K && bx * BN + c < N) ? 16 : 0;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm_addr), "l"(g_ptr), "r"(pred));
        }
    };

    load_to_stage(0, 0); asm volatile("cp.async.commit_group;\n");
    if (BK < K) load_to_stage(1, BK); asm volatile("cp.async.commit_group;\n");
    
    int write_stage = 2, read_stage = 0;
    uint32_t frA[2][4][4], frB[2][8][2];

    asm volatile("cp.async.wait_group 1;\n"); __syncthreads();

    auto load_frA = [&](uint32_t reg[4], int ki, int mi, int st) {
        int r = wy * 64 + mi * 16 + ((lane / 8) % 2) * 8 + (lane % 8);
        int c = ki + (lane / 16) * 8;
        int swizzled_c = c ^ ((r & 3) << 3);
        uint32_t addr = __cvta_generic_to_shared(&s_mem[st * STAGE_SIZE + r * BK + swizzled_c]);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];" 
                     : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3]) : "r"(addr));
    };

    auto load_frB = [&](uint32_t reg[2], int ki, int ni, int st) {
        int r = ki + ((lane / 8) % 2) * 8 + (lane % 8);
        int c = wx * 64 + ni * 8;
        int swizzled_c = c ^ ((r & 7) << 3);
        uint32_t addr = __cvta_generic_to_shared(&s_mem[st * STAGE_SIZE + AS_SIZE + r * BN + swizzled_c]);
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];" 
                     : "=r"(reg[0]), "=r"(reg[1]) : "r"(addr));
    };

    #pragma unroll
    for (int i = 0; i < 4; i++) load_frA(frA[0][i], 0, i, read_stage);
    #pragma unroll
    for (int j = 0; j < 8; j++) load_frB(frB[0][j], 0, j, read_stage);

    for (int k = 0; k < K; k += BK) {
        if (k + 2 * BK < K) load_to_stage(write_stage, k + 2 * BK); 
        asm volatile("cp.async.commit_group;\n");

        #pragma unroll
        for (int ks = 0; ks < 32; ks += 16) {
            int p = (ks >> 4) & 1, q = p ^ 1;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                if (ks + 16 < 32) {
                    load_frA(frA[q][i], ks + 16, i, read_stage);
                    load_frB(frB[q][i*2], ks + 16, i*2, read_stage);
                    load_frB(frB[q][i*2+1], ks + 16, i*2+1, read_stage);
                } else if (k + BK < K) {
                    if (i == 0) {
                        asm volatile("cp.async.wait_group 1;\n");
                        __syncthreads();
                        read_stage = (read_stage + 1) % STAGES;
                        write_stage = (write_stage + 1) % STAGES;
                    }
                    load_frA(frA[q][i], 0, i, read_stage);
                    load_frB(frB[q][i*2], 0, i*2, read_stage);
                    load_frB(frB[q][i*2+1], 0, i*2+1, read_stage);
                }
                #pragma unroll
                for (int j = 0; j < 8; j++) MMA_M16N8K16_F32(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3], frA[p][i][0], frA[p][i][1], frA[p][i][2], frA[p][i][3], frB[p][j][0], frB[p][j][1]);
            }
        }
    }

    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    // Final Vectorized Epilogue
    const int laneId = lane, g = laneId / 4, t = laneId % 4;
    float alpha_f = __half2float(alpha);
    float beta_f = __half2float(beta);

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float acc0 = acc[i][j][0] * alpha_f; float acc1 = acc[i][j][1] * alpha_f;
            float acc2 = acc[i][j][2] * alpha_f; float acc3 = acc[i][j][3] * alpha_f;

            float n0 = __shfl_xor_sync(0xffffffff, acc0, 1);
            float n1 = __shfl_xor_sync(0xffffffff, acc1, 1);
            float n2 = __shfl_xor_sync(0xffffffff, acc2, 1);
            float n3 = __shfl_xor_sync(0xffffffff, acc3, 1);

            float4 f4_r0, f4_r8;
            if (t % 2 == 0) { f4_r0 = {acc0, acc1, n0, n1}; f4_r8 = {acc2, acc3, n2, n3}; }
            else           { f4_r0 = {n0, n1, acc0, acc1}; f4_r8 = {n2, n3, acc2, acc3}; }

            if (t % 2 == 0) {
                const int r0 = by * BM + wy * 64 + i * 16 + g, r8 = r0 + 8;
                const int c = bx * BN + wx * 64 + j * 8 + (t / 2) * 4;
                
                float4 bv0 = {0,0,0,0}, bv8 = {0,0,0,0};
                if (bias) {
                    if (bias_numel == 1) { float b = __half2float(bias[0]); bv0 = bv8 = {b,b,b,b}; }
                    else if (bias_numel == N) {
                         if (c < N) {
                             if (c + 7 < N && (((size_t)(&bias[c]) & 15) == 0)) {
                                 // Vectorized load of 8 halfs (16 bytes)
                                 uint4 b_vec = *(uint4*)&bias[c];
                                 __half* bh = (__half*)&b_vec;
                                 bv0 = {__half2float(bh[0]), __half2float(bh[1]), __half2float(bh[2]), __half2float(bh[3])};
                                 bv8 = bv0; // Row-independent bias
                             } else {
                                 bv0.x = __half2float(bias[c]); 
                                 if(c+1<N) bv0.y = __half2float(bias[c+1]); 
                                 if(c+2<N) bv0.z = __half2float(bias[c+2]); 
                                 if(c+3<N) bv0.w = __half2float(bias[c+3]);
                                 bv8 = bv0;
                             }
                         }
                    } else if (bias_numel == (int64_t)M * N) {
                         const int64_t idx0 = (int64_t)r0 * N + c, idx8 = (int64_t)r8 * N + c;
                         if (r0 < M && c < N) {
                             bv0.x = __half2float(bias[idx0]); if(c+1<N) bv0.y = __half2float(bias[idx0+1]); if(c+2<N) bv0.z = __half2float(bias[idx0+2]); if(c+3<N) bv0.w = __half2float(bias[idx0+3]);
                         }
                         if (r8 < M && c < N) {
                             bv8.x = __half2float(bias[idx8]); if(c+1<N) bv8.y = __half2float(bias[idx8+1]); if(c+2<N) bv8.z = __half2float(bias[idx8+2]); if(c+3<N) bv8.w = __half2float(bias[idx8+3]);
                         }
                    }
                }
                
                f4_r0.x += beta_f * bv0.x; f4_r0.y += beta_f * bv0.y; f4_r0.z += beta_f * bv0.z; f4_r0.w += beta_f * bv0.w;
                f4_r8.x += beta_f * bv8.x; f4_r8.y += beta_f * bv8.y; f4_r8.z += beta_f * bv8.z; f4_r8.w += beta_f * bv8.w;

                auto sh4 = [&](int r, int cl, float4 v) {
                    if (r < M && cl < N) {
                        __half* p = C_batch + (long long)r * ldc + cl;
                        // For half, we can't easily use float4 for storing 4 halfs if not aligned to 8 bytes.
                        // But each thread writes 4 consecutive elements.
                        p[0] = __float2half(v.x);
                        if(cl+1<N) p[1] = __float2half(v.y);
                        if(cl+2<N) p[2] = __float2half(v.z);
                        if(cl+3<N) p[3] = __float2half(v.w);
                    }
                };
                sh4(r0, c, f4_r0); sh4(r8, c, f4_r8);
            }
        }
    }
}

extern "C" void mycublasHgemmAddmm_v34(
    mycublasHandle_t handle, int M, int N, int K, const __half alpha,
    const __half* d_A, int lda, long long int strideA,
    const __half* d_B, int ldb, long long int strideB,
    const __half beta, 
    const __half* d_bias, int64_t bias_numel,
    __half* d_C, int ldc, long long int strideC, int batchCount)
{
    cudaStream_t stream = (handle != nullptr) ? handle->stream : 0;
    static const size_t smem_bytes = STAGES * STAGE_SIZE * sizeof(__half);
    
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(hgemm_addmm_v34_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
        cudaFuncSetAttribute(hgemm_addmm_v34_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
        attr_set = true;
    }

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, batchCount);
    // For now we don't use the IsAligned template parameter as extensively as FP32 version yet, 
    // but we keep the structure for future optimization.
    hgemm_addmm_v34_kernel<true><<<grid, THREADS, smem_bytes, stream>>>(M,N,K,alpha,d_A,lda,strideA,d_B,ldb,strideB,beta,d_bias,bias_numel,d_C,ldc,strideC,batchCount);
}
