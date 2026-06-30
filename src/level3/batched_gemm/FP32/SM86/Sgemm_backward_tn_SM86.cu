#include "mycublas.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <algorithm>
#include <unordered_map>

// ============================================================
// Optimized Backward SGEMM TN Kernel V18
// dB = dX^T dY (A: [K, M], B: [K, N], C: [M, N])
// Mapping: C = A^T * B
// Optimizations: 16-wide Swizzling, 3-stage Async, SplitK
// ============================================================

#define BM      128
#define BN      128
#define BK      16
#define STAGES  3
#define THREADS 128
#define AS_SIZE    (BK * BM)  
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

template <bool IsAligned, int SplitK>
__global__ void __launch_bounds__(THREADS, 1)
sgemm_tn_backward_kernel_SM86(
    int M, int N, int K,
    float alpha,
    const float* __restrict__ A, int lda, long long strideA,
    const float* __restrict__ B, int ldb, long long strideB,
    float beta,
    float* __restrict__ C, int ldc, long long strideC,
    int batchCount)
{
    const int batch = blockIdx.z / SplitK;
    const int sk_idx = blockIdx.z % SplitK;
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
    const int wy = (wid >> 1), wx = (wid & 1);

    const int k_tiles = (K + BK - 1) / BK;
    const int tiles_per_sk = (k_tiles + SplitK - 1) / SplitK;
    const int k_start = sk_idx * tiles_per_sk * BK;
    const int k_end = min(K, (sk_idx + 1) * tiles_per_sk * BK);

    // For TN: C[M, N] = A[K, M]^T * B[K, N]
    // Inner dimension is K.
    const float* gA_ptr[4];
    const float* gB_ptr[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int r = (tid / 32) + i * 4, c = (tid % 32) * 4;
        gA_ptr[i] = A + (long long)batch * strideA + (long long)(k_start + r) * lda + (by * BM + c);
        gB_ptr[i] = B + (long long)batch * strideB + (long long)(k_start + r) * ldb + (bx * BN + c);
    }

    extern __shared__ float smem[];
    float acc[4][8][4];
    #pragma unroll
    for (int i = 0; i < 4; i++) for (int j = 0; j < 8; j++) for (int d = 0; d < 4; d++) acc[i][j][d] = 0.f;

    auto load_to_stage = [&](int stage, int ko) {
        float* As = smem + stage * STAGE_SIZE;
        float* Bs = As + AS_SIZE;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int r = (tid / 32) + i * 4, c = (tid % 32) * 4, sc = c ^ ((r & 7) << 3);
            uint32_t sm_a = __cvta_generic_to_shared(&As[r * BM + sc]);
            int gk = ko + r, gm = by * BM + c;
            if constexpr (IsAligned) {
                int bytes = (gk < K && gm < M) ? max(0, min(16, (M - gm) * 4)) : 0;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm_a), "l"(gA_ptr[i]), "r"(bytes));
            } else {
                float4 val = {0,0,0,0};
                if (gk < K) {
                    if (gm < M) val.x = gA_ptr[i][0];
                    if (gm + 1 < M) val.y = gA_ptr[i][1];
                    if (gm + 2 < M) val.z = gA_ptr[i][2];
                    if (gm + 3 < M) val.w = gA_ptr[i][3];
                }
                *(float4*)&As[r * BM + sc] = val;
            }
            gA_ptr[i] += BK * lda;
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int r = (tid / 32) + i * 4, c = (tid % 32) * 4, sc = c ^ ((r & 7) << 3);
            uint32_t sm_b = __cvta_generic_to_shared(&Bs[r * BN + sc]);
            int gk = ko + r, gn = bx * BN + c;
            if constexpr (IsAligned) {
                int bytes = (gk < K && gn < N) ? max(0, min(16, (N - gn) * 4)) : 0;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm_b), "l"(gB_ptr[i]), "r"(bytes));
            } else {
                float4 val = {0,0,0,0};
                if (gk < K) {
                    if (gn < N) val.x = gB_ptr[i][0];
                    if (gn + 1 < N) val.y = gB_ptr[i][1];
                    if (gn + 2 < N) val.z = gB_ptr[i][2];
                    if (gn + 3 < N) val.w = gB_ptr[i][3];
                }
                *(float4*)&Bs[r * BN + sc] = val;
            }
            gB_ptr[i] += BK * ldb;
        }
    };

    const int g_sh = lane / 4, t_sh = lane % 4;
    int m_base[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) m_base[i] = (wy * 64 + i * 16 + g_sh);

    auto load_frA = [&](uint32_t reg[4], int ks, int mi, int st) {
        float* As = smem + st * STAGE_SIZE;
        const int k0 = ks + t_sh, k4 = k0 + 4, row = m_base[mi];
        auto ga = [&](int k, int m) { return *(uint32_t*)&As[k * BM + (m ^ ((k & 7) << 3))]; };
        reg[0] = ga(k0, row); reg[1] = ga(k0, row + 8);
        reg[2] = ga(k4, row); reg[3] = ga(k4, row + 8);
    };

    int n_base[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) n_base[j] = (wx * 64 + j * 8 + g_sh);

    auto load_frB = [&](uint32_t reg[2], int ks, int ni, int st) {
        float* Bs = smem + st * STAGE_SIZE + AS_SIZE;
        const int k0 = ks + t_sh, k4 = k0 + 4, col = n_base[ni];
        auto gb = [&](int k, int n) { return *(uint32_t*)&Bs[k * BN + (n ^ ((k & 7) << 3))]; };
        reg[0] = gb(k0, col); reg[1] = gb(k4, col);
    };

    if (k_start < k_end) {
        load_to_stage(0, k_start); asm volatile("cp.async.commit_group;\n");
        if (k_start + BK < k_end) { load_to_stage(1, k_start + BK); asm volatile("cp.async.commit_group;\n"); }
    }

    int ws = 2, rs = 0; uint32_t frA[2][4][4], frB[2][8][2];
    asm volatile("cp.async.wait_group 1;\n"); __syncthreads();
    for (int i = 0; i < 4; i++) load_frA(frA[0][i], 0, i, rs);
    for (int j = 0; j < 8; j++) load_frB(frB[0][j], 0, j, rs);

    for (int k = k_start; k < k_end; k += BK) {
        if (k + 2 * BK < k_end) { load_to_stage(ws, k + 2 * BK); asm volatile("cp.async.commit_group;\n"); }
        #pragma unroll
        for (int ks = 0; ks < BK; ks += 8) {
            const int p = (ks / 8) & 1, q = p ^ 1;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                if (ks + 8 < BK) { 
                    load_frA(frA[q][i], ks + 8, i, rs);
                    load_frB(frB[q][i*2], ks + 8, i*2, rs);
                    load_frB(frB[q][i*2+1], ks + 8, i*2+1, rs);
                } else if (k + BK < k_end) { 
                    if (i == 0) { asm volatile("cp.async.wait_group 1;\n"); __syncthreads(); rs = (rs+1)%STAGES; ws = (ws+1)%STAGES; }
                    load_frA(frA[q][i], 0, i, rs);
                    load_frB(frB[q][i*2], 0, i*2, rs);
                    load_frB(frB[q][i*2+1], 0, i*2+1, rs);
                }
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
            float* dC = C + (long long)batch * strideC;
            auto sf2 = [&](int r, int c, float f0, float f1) {
                if (r >= M || c >= N) return;
                float* dst = &dC[(long long)r * ldc + c];
                if constexpr (SplitK > 1) {
                    atomicAdd(dst, alpha * f0);
                    if (c + 1 < N) atomicAdd(dst + 1, alpha * f1);
                } else {
                    if (c + 1 < N && (((size_t)dst & 7) == 0)) {
                        float2 old = (beta == 0.f) ? make_float2(0,0) : *(float2*)dst;
                        *(float2*)dst = { alpha * f0 + beta * old.x, alpha * f1 + beta * old.y };
                    } else {
                        dst[0] = alpha * f0 + (beta == 0 ? 0 : beta * dst[0]);
                        if (c + 1 < N) dst[1] = alpha * f1 + (beta == 0 ? 0 : beta * dst[1]);
                    }
                }
            };
            sf2(r0, col, acc[i][j][0], acc[i][j][1]); sf2(r8, col, acc[i][j][2], acc[i][j][3]);
        }
    }
}

__global__ void sgemm_tn_scale_kernel_SM86(float* C, float beta, int M, int N, int ldc, long long strideC, int batchCount) {
    int r = blockIdx.y * blockDim.y + threadIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x, b = blockIdx.z;
    if (r < M && c < N && b < batchCount) {
        float* dst = &C[b * strideC + (long long)r * ldc + c];
        *dst = (beta == 0.f) ? 0.f : (*dst * beta);
    }
}

extern "C" void mycublasSgemmStridedBatched_tn_SM86(
    mycublasHandle_t handle, int M, int N, int K, const float alpha,
    const float* d_A, int lda, long long int strideA,
    const float* d_B, int ldb, long long int strideB,
    const float beta, float* d_C, int ldc, long long int strideC,
    int batchCount)
{
    cudaStream_t stream = (handle != nullptr) ? handle->stream : 0;

    static int cached_sm_count = 0;
    static int cached_sm_ver   = 0;
    if (cached_sm_count == 0) {
        int dev; cudaGetDevice(&dev);
        cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
        cached_sm_count = prop.multiProcessorCount;
        cached_sm_ver   = prop.major * 10 + prop.minor;
    }
    const int sm_count = cached_sm_count;
    const int sm_ver   = cached_sm_ver;

    // Tier 1: GEMV
    if (M <= 16 || N <= 16) {
        mycublasSgemv_tn_SM86(handle, M, N, K, alpha,
                             d_A, lda, strideA, d_B, ldb, strideB,
                             beta, d_C, ldc, strideC, batchCount);
        return;
    }

    const int gx = (N+BN-1)/BN, gy = (M+BM-1)/BM;

    // Tier 2: 64x64 tile — improved occupancy for medium shapes, scaled by SM count
    if (gx * gy < sm_count) {
        
            mycublasSgemmStridedBatched_tn_tile64_SM86(handle, M, N, K, alpha,
                                                      d_A, lda, strideA, d_B, ldb, strideB,
                                                      beta, d_C, ldc, strideC, batchCount);
        return;
    }

    static const size_t smem_bytes = STAGES * STAGE_SIZE * sizeof(float);
    const bool aligned = (((size_t)d_A & 15) == 0) && ((lda & 3) == 0) && (((size_t)d_B & 15) == 0) && ((ldb & 3) == 0);
    auto set_limit = [](const void* f, size_t b) {
        static thread_local std::unordered_map<const void*, size_t> cache;
        if (cache.find(f) == cache.end() || cache[f] < b) { cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)b); cache[f] = b; }
    };

    // TN uses sk=16 max. Higher sk causes L2 atomic contention on small output tiles.
    int sk = 1; if (gx * gy < sm_count * 2 && K >= 1024) { sk = 4; if (gx * gy < sm_count / 2) sk = 8; if (gx * gy < sm_count / 8) sk = 16; }
    if (sk > 1) { dim3 s_grid((N+31)/32, (M+31)/32, batchCount); sgemm_tn_scale_kernel_SM86<<<s_grid, dim3(32,32), 0, stream>>>(d_C, beta, M, N, ldc, strideC, batchCount); }

    dim3 grid(gx, gy, batchCount * sk);
    if (aligned) {
        if (sk == 1) { set_limit((const void*)sgemm_tn_backward_kernel_SM86<true, 1>, smem_bytes); sgemm_tn_backward_kernel_SM86<true, 1><<<grid, THREADS, smem_bytes, stream>>>(M,N,K,alpha,d_A,lda,strideA,d_B,ldb,strideB,beta,d_C,ldc,strideC,batchCount); }
        else if (sk == 4) { set_limit((const void*)sgemm_tn_backward_kernel_SM86<true, 4>, smem_bytes); sgemm_tn_backward_kernel_SM86<true, 4><<<grid, THREADS, smem_bytes, stream>>>(M,N,K,alpha,d_A,lda,strideA,d_B,ldb,strideB,0.f,d_C,ldc,strideC,batchCount); }
        else if (sk == 8) { set_limit((const void*)sgemm_tn_backward_kernel_SM86<true, 8>, smem_bytes); sgemm_tn_backward_kernel_SM86<true, 8><<<grid, THREADS, smem_bytes, stream>>>(M,N,K,alpha,d_A,lda,strideA,d_B,ldb,strideB,0.f,d_C,ldc,strideC,batchCount); }
        else { set_limit((const void*)sgemm_tn_backward_kernel_SM86<true, 16>, smem_bytes); sgemm_tn_backward_kernel_SM86<true, 16><<<grid, THREADS, smem_bytes, stream>>>(M,N,K,alpha,d_A,lda,strideA,d_B,ldb,strideB,0.f,d_C,ldc,strideC,batchCount); }
    } else {
        if (sk == 1) { set_limit((const void*)sgemm_tn_backward_kernel_SM86<false, 1>, smem_bytes); sgemm_tn_backward_kernel_SM86<false, 1><<<grid, THREADS, smem_bytes, stream>>>(M,N,K,alpha,d_A,lda,strideA,d_B,ldb,strideB,beta,d_C,ldc,strideC,batchCount); }
        else if (sk == 4) { set_limit((const void*)sgemm_tn_backward_kernel_SM86<false, 4>, smem_bytes); sgemm_tn_backward_kernel_SM86<false, 4><<<grid, THREADS, smem_bytes, stream>>>(M,N,K,alpha,d_A,lda,strideA,d_B,ldb,strideB,0.f,d_C,ldc,strideC,batchCount); }
        else if (sk == 8) { set_limit((const void*)sgemm_tn_backward_kernel_SM86<false, 8>, smem_bytes); sgemm_tn_backward_kernel_SM86<false, 8><<<grid, THREADS, smem_bytes, stream>>>(M,N,K,alpha,d_A,lda,strideA,d_B,ldb,strideB,0.f,d_C,ldc,strideC,batchCount); }
        else { set_limit((const void*)sgemm_tn_backward_kernel_SM86<false, 16>, smem_bytes); sgemm_tn_backward_kernel_SM86<false, 16><<<grid, THREADS, smem_bytes, stream>>>(M,N,K,alpha,d_A,lda,strideA,d_B,ldb,strideB,0.f,d_C,ldc,strideC,batchCount); }
    }
}