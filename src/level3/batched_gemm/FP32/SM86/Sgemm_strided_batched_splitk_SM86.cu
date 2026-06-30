#include "mycublas.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include <algorithm>
#include <unordered_map>

// ============================================================
// Sgemm V18 — SplitK Edition
//
// NN layout (A row-major, B row-major): C = A * B
// SplitK splits the K dimension across blockIdx.z, then
// atomicAdds partial results into C (when SplitK > 1).
// A separate scale kernel pre-applies beta before the adds.
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

template <bool IsAligned, int SplitK>
__global__ void __launch_bounds__(THREADS, 1)
sgemm_SM86_splitk_kernel(
    int M, int N, int K,
    float alpha,
    const float* __restrict__ A, int lda, long long int strideA,
    const float* __restrict__ B, int ldb, long long int strideB,
    float beta,
    float* __restrict__ C, int ldc, long long int strideC,
    int batchCount)
{
    const int batch  = (int)blockIdx.z / SplitK;
    const int sk_idx = (int)blockIdx.z % SplitK;
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

    // SplitK range for this block
    const int k_tiles      = (K + BK - 1) / BK;
    const int tiles_per_sk = (k_tiles + SplitK - 1) / SplitK;
    const int k_start      = sk_idx * tiles_per_sk * BK;
    const int k_end        = min(K, (sk_idx + 1) * tiles_per_sk * BK);

    // Pointer Induction: Initial Global Pointers (offset by k_start)
    const float* gA_ptr[4];
    const float* gB_ptr[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const int rowA = by * BM + (tid / 4) + i * 32;
        gA_ptr[i] = A + (long long)batch * strideA + (long long)rowA * lda + (k_start + (tid % 4) * 4);
        const int rowB = i * 4 + (tid / 32);
        gB_ptr[i] = B + (long long)batch * strideB + (long long)(k_start + rowB) * ldb + (bx * BN + (tid % 32) * 4);
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

    if (k_start < k_end) {
        load_to_stage(0, k_start); asm volatile("cp.async.commit_group;\n");
        if (k_start + BK < k_end) load_to_stage(1, k_start + BK); asm volatile("cp.async.commit_group;\n");
    }

    int write_stage = 2, read_stage = 0; uint32_t frA[2][4][4], frB[2][8][2];
    asm volatile("cp.async.wait_group 1;\n"); __syncthreads();
    for (int i = 0; i < 4; i++) load_frA(frA[0][i], 0, i, read_stage);
    for (int j = 0; j < 8; j++) load_frB(frB[0][j], 0, j, read_stage);

    for (int k = k_start; k < k_end; k += BK) {
        if (k + 2 * BK < k_end) load_to_stage(write_stage, k + 2 * BK); asm volatile("cp.async.commit_group;\n");
        #pragma unroll
        for (int ks = 0; ks < BK; ks += 8) {
            const int p = (ks / 8) & 1, q = p ^ 1;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                if (ks + 8 < BK) { load_frA(frA[q][i], ks + 8, i, read_stage); load_frB(frB[q][i * 2], ks + 8, i * 2, read_stage); load_frB(frB[q][i * 2 + 1], ks + 8, i * 2 + 1, read_stage); }
                else if (k + BK < k_end) { if (i == 0) { asm volatile("cp.async.wait_group 1;\n"); __syncthreads(); read_stage = (read_stage+1)%STAGES; write_stage = (write_stage+1)%STAGES; }
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
                if constexpr (SplitK > 1) {
                    atomicAdd(dst,     alpha * f0);
                    if (c + 1 < N) atomicAdd(dst + 1, alpha * f1);
                } else {
                    if (c + 1 < N && (((size_t)dst & 7) == 0)) { float2 old = (beta == 0.f) ? make_float2(0,0) : *(float2*)dst; *(float2*)dst = { alpha * f0 + beta * old.x, alpha * f1 + beta * old.y }; }
                    else { dst[0] = alpha * f0 + (beta == 0 ? 0 : beta * dst[0]); if (c + 1 < N) dst[1] = alpha * f1 + (beta == 0 ? 0 : beta * dst[1]); }
                }
            };
            sf2(r0, col, acc[i][j][0], acc[i][j][1]); sf2(r8, col, acc[i][j][2], acc[i][j][3]);
        }
    }
}

// Scale C by beta (or zero it) before SplitK atomic accumulation
__global__ void sgemm_splitk_scale_kernel_SM86(
    float* C, float beta, int M, int N, int ldc, long long strideC, int batchCount)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.z;
    if (r < M && c < N && b < batchCount) {
        float* dst = &C[(long long)b * strideC + (long long)r * ldc + c];
        *dst = (beta == 0.f) ? 0.f : (*dst * beta);
    }
}

extern "C" void mycublasSgemmStridedBatched_splitk_SM86(
    mycublasHandle_t handle, int M, int N, int K, const float alpha,
    const float* d_A, int lda, long long int strideA,
    const float* d_B, int ldb, long long int strideB,
    const float beta, float* d_C, int ldc, long long int strideC, int batchCount)
{
    cudaStream_t stream = (handle != nullptr) ? handle->stream : 0;
    static const size_t smem_bytes = STAGES * STAGE_SIZE * sizeof(float);
    const bool aligned = (((size_t)d_A & 15) == 0) && ((lda & 3) == 0) &&
                         (((size_t)d_B & 15) == 0) && ((ldb & 3) == 0);

    auto set_limit = [](const void* f, size_t b) {
        static thread_local std::unordered_map<const void*, size_t> cache;
        if (cache.find(f) == cache.end() || cache[f] < b) {
            cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)b);
            cache[f] = b;
        }
    };

    // Choose SplitK factor based on grid size and K depth.
    // Higher sk means more blocks share the K dimension via atomicAdd.
    // Goal: ensure total blocks >= ~0.75 * SM_count so all SMs stay busy.
    const int gx = (N + BN - 1) / BN, gy = (M + BM - 1) / BM;
    int sk = 1;
    if (gx * gy < 160 && K >= 1024) {
        sk = 4;
        if (gx * gy < 40)  sk = 8;
        if (gx * gy < 10)  sk = 16;
        if (gx * gy < 4)   sk = 32;
        if (gx * gy == 1 && K >= 2048) sk = 64;
    }

    // Pre-scale C by beta before atomicAdds when SplitK > 1
    if (sk > 1) {
        dim3 s_grid((N + 31) / 32, (M + 31) / 32, batchCount);
        sgemm_splitk_scale_kernel_SM86<<<s_grid, dim3(32, 32), 0, stream>>>(
            d_C, beta, M, N, ldc, strideC, batchCount);
    }

    dim3 grid(gx, gy, batchCount * sk);

#define DISPATCH(aligned_val, sk_val)                                                                 \
    set_limit((const void*)sgemm_SM86_splitk_kernel<aligned_val, sk_val>, smem_bytes);                \
    sgemm_SM86_splitk_kernel<aligned_val, sk_val><<<grid, THREADS, smem_bytes, stream>>>(             \
        M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB,                                        \
        (sk_val > 1 ? 0.f : beta), d_C, ldc, strideC, batchCount)

    if (aligned) {
        if      (sk ==  1) { DISPATCH(true,  1); }
        else if (sk ==  4) { DISPATCH(true,  4); }
        else if (sk ==  8) { DISPATCH(true,  8); }
        else if (sk == 16) { DISPATCH(true, 16); }
        else if (sk == 32) { DISPATCH(true, 32); }
        else               { DISPATCH(true, 64); }
    } else {
        if      (sk ==  1) { DISPATCH(false,  1); }
        else if (sk ==  4) { DISPATCH(false,  4); }
        else if (sk ==  8) { DISPATCH(false,  8); }
        else if (sk == 16) { DISPATCH(false, 16); }
        else if (sk == 32) { DISPATCH(false, 32); }
        else               { DISPATCH(false, 64); }
    }
#undef DISPATCH
}
