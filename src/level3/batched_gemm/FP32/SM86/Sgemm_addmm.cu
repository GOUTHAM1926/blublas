
#include "mycublas.h"
#include <cuda_runtime.h>
#include <stdint.h>

// ============================================================
// Sgemm Addmm V34 — Fused Matmul + Bias (v18 Base)
//
// Highlights:
//   1. Derived from v18 Ultimate (Revision 7)
//   2. Fused Bias addition in Epilogue
//   3. Supports scalar and vector bias broadcasting
// ============================================================

#define BM      128
#define BN      128
#define BK      16
#define STAGES  3
#define THREADS 128
#define AS_SIZE    (BM * BK)
#define BS_SIZE    (BK * BN)
#define STAGE_SIZE (AS_SIZE + BS_SIZE)

#ifndef MMA_TF32
#define MMA_TF32(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1)                \
    asm volatile(                                                   \
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "     \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"       \
        : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)                     \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1))
#endif

template <bool IsAligned>
__global__ void __launch_bounds__(THREADS, 1)
sgemm_addmm_kernel(
    int M, int N, int K,
    float alpha,
    const float* __restrict__ A, int lda, long long strideA,
    const float* __restrict__ B, int ldb, long long strideB,
    float beta,
    const float* __restrict__ bias, int64_t bias_numel,
    float* __restrict__ C, int ldc, long long strideC,
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

    // Final Vectorized Epilogue (High-Performance)
    const int laneId = lane, g = laneId / 4, t = laneId % 4;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float acc0 = acc[i][j][0] * alpha; float acc1 = acc[i][j][1] * alpha;
            float acc2 = acc[i][j][2] * alpha; float acc3 = acc[i][j][3] * alpha;

            // Combine column pairs from adjacent threads (t=0,1 and t=2,3)
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
                float* dst_base = C + (long long)batch * strideC;

                float4 bv0 = {0,0,0,0}, bv8 = {0,0,0,0};
                if (bias) {
                    if (bias_numel == 1) { float b = bias[0]; bv0 = bv8 = {b,b,b,b}; }
                    else if (bias_numel == N) {
                         if (c < N) {
                             if (c + 3 < N && (((size_t)(&bias[c]) & 15) == 0)) bv0 = *(float4*)&bias[c];
                             else { bv0.x=bias[c]; if(c+1<N) bv0.y=bias[c+1]; if(c+2<N) bv0.z=bias[c+2]; if(c+3<N) bv0.w=bias[c+3]; }
                             bv8 = bv0;
                         }
                    } else if (bias_numel == (int64_t)M * N) {
                         const int64_t idx0 = (int64_t)r0 * N + c, idx8 = (int64_t)r8 * N + c;
                         if (r0 < M && c < N) {
                             if (c + 3 < N && (((size_t)(&bias[idx0]) & 15) == 0)) bv0 = *(float4*)&bias[idx0];
                             else { bv0.x=bias[idx0]; if(c+1<N) bv0.y=bias[idx0+1]; if(c+2<N) bv0.z=bias[idx0+2]; if(c+3<N) bv0.w=bias[idx0+3]; }
                         }
                         if (r8 < M && c < N) {
                             if (c + 3 < N && (((size_t)(&bias[idx8]) & 15) == 0)) bv8 = *(float4*)&bias[idx8];
                             else { bv8.x=bias[idx8]; if(c+1<N) bv8.y=bias[idx8+1]; if(c+2<N) bv8.z=bias[idx8+2]; if(c+3<N) bv8.w=bias[idx8+3]; }
                         }
                    }
                }
                
                f4_r0.x += beta * bv0.x; f4_r0.y += beta * bv0.y; f4_r0.z += beta * bv0.z; f4_r0.w += beta * bv0.w;
                f4_r8.x += beta * bv8.x; f4_r8.y += beta * bv8.y; f4_r8.z += beta * bv8.z; f4_r8.w += beta * bv8.w;

                auto sf4 = [&](int r, int cl, float4 v) {
                    if (r < M && cl < N) {
                        float* p = dst_base + (long long)r * ldc + cl;
                        if (cl + 3 < N && (((size_t)p & 15) == 0)) { *(float4*)p = v; }
                        else { p[0]=v.x; if(cl+1<N) p[1]=v.y; if(cl+2<N) p[2]=v.z; if(cl+3<N) p[3]=v.w; }
                    }
                };
                sf4(r0, c, f4_r0); sf4(r8, c, f4_r8);
            }
        }
    }
}

extern "C" void mycublasSgemmAddmm_SM86(
    mycublasHandle_t handle, int M, int N, int K, const float alpha,
    const float* d_A, int lda, long long int strideA,
    const float* d_B, int ldb, long long int strideB,
    const float beta, 
    const float* d_bias, int64_t bias_numel,
    float* d_C, int ldc, long long int strideC, int batchCount)
{
    cudaStream_t stream = (handle != nullptr) ? handle->stream : 0;
    const bool aligned = (((size_t)d_A & 15) == 0) && ((lda & 3) == 0) && (((size_t)d_B & 15) == 0) && ((ldb & 3) == 0);
    static const size_t smem_bytes = STAGES * STAGE_SIZE * sizeof(float);
    
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(sgemm_addmm_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        cudaFuncSetAttribute(sgemm_addmm_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        attr_set = true;
    }

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, batchCount);
    if (aligned) sgemm_addmm_kernel<true><<<grid, THREADS, smem_bytes, stream>>>(M,N,K,alpha,d_A,lda,strideA,d_B,ldb,strideB,beta,d_bias,bias_numel,d_C,ldc,strideC,batchCount);
    else sgemm_addmm_kernel<false><<<grid, THREADS, smem_bytes, stream>>>(M,N,K,alpha,d_A,lda,strideA,d_B,ldb,strideB,beta,d_bias,bias_numel,d_C,ldc,strideC,batchCount);
}
