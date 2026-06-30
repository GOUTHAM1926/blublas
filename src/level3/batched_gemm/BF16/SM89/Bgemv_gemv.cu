#include "mycublas.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// ============================================================
// GEMV Dispatch Kernels BF16 (Bgemv) - Vectorized
// ============================================================

#define GEMV_THREADS  256
#define GEMV_BK       256

// NN M==1: C[n] = alpha * sum(A[k] * B[k,n])
template<bool Beta0>
__global__ void bgemv_nn_row_vec8_kernel(
    int N, int K, __nv_bfloat16 alpha,
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B, int ldb,
    __nv_bfloat16 beta, __nv_bfloat16* __restrict__ C,
    int batchCount, long long sA, long long sB, long long sC)
{
    const int b = blockIdx.z; if (b >= batchCount) return;
    const __nv_bfloat16* gA = A + b * sA;
    const __nv_bfloat16* gB = B + b * sB;
    __nv_bfloat16*       gC = C + b * sC;
    
    const int n_start = (blockIdx.x * GEMV_THREADS + threadIdx.x) * 8;
    if (n_start >= N) return;

    __shared__ __nv_bfloat16 smA[GEMV_BK];
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int k0 = 0; k0 < K; k0 += GEMV_BK) {
        int klen = min(GEMV_BK, K - k0);
        if (threadIdx.x < klen) smA[threadIdx.x] = gA[k0 + threadIdx.x];
        __syncthreads();

        for (int ki = 0; ki < klen; ki++) {
            __nv_bfloat16 a_val = smA[ki];
            const float4* b_ptr = (const float4*)&gB[(long long)(k0 + ki) * ldb + n_start];
            float4 b_vec = *b_ptr;
            const __nv_bfloat162* b_bf2 = (const __nv_bfloat162*)&b_vec;
            
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                acc[i*2]   += __bfloat162float(a_val) * __bfloat162float(b_bf2[i].x);
                acc[i*2+1] += __bfloat162float(a_val) * __bfloat162float(b_bf2[i].y);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (n_start + i < N) {
            float res = __bfloat162float(alpha) * acc[i];
            if constexpr (!Beta0) res += __bfloat162float(beta) * __bfloat162float(gC[n_start + i]);
            gC[n_start + i] = __float2bfloat16(res);
        }
    }
}

// NT M==1: C[n] = alpha * sum(A[k] * B[n, k])
template<bool Beta0>
__global__ void bgemv_nt_row_kernel(
    int N, int K, __nv_bfloat16 alpha,
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B, int ldb,
    __nv_bfloat16 beta, __nv_bfloat16* __restrict__ C,
    int batchCount, long long sA, long long sB, long long sC)
{
    const int b = blockIdx.z; if (b >= batchCount) return;
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int n = blockIdx.x * (GEMV_THREADS / 32) + warp;
    if (n >= N) return;

    const __nv_bfloat16* gA = A + b * sA;
    const __nv_bfloat16* gB = B + b * sB + (long long)n * ldb;
    __nv_bfloat16*       gC = C + b * sC;

    float acc = 0.f;
    const float4* a_vec_ptr = (const float4*)gA;
    const float4* b_vec_ptr = (const float4*)gB;
    
    int k_vec_limit = K / 8;
    for (int k0 = lane; k0 < k_vec_limit; k0 += 32) {
        float4 av = a_vec_ptr[k0];
        float4 bv = b_vec_ptr[k0];
        const __nv_bfloat162* ah2 = (const __nv_bfloat162*)&av;
        const __nv_bfloat162* bh2 = (const __nv_bfloat162*)&bv;
        #pragma unroll
        for(int i=0; i<4; i++) {
            acc += __bfloat162float(ah2[i].x) * __bfloat162float(bh2[i].x);
            acc += __bfloat162float(ah2[i].y) * __bfloat162float(bh2[i].y);
        }
    }
    for (int k = k_vec_limit * 8 + lane; k < K; k += 32) {
        acc += __bfloat162float(gA[k]) * __bfloat162float(gB[k]);
    }

    for (int offset = 16; offset > 0; offset /= 2)
        acc += __shfl_down_sync(0xffffffff, acc, offset);

    if (lane == 0) {
        float res = __bfloat162float(alpha) * acc;
        if constexpr (!Beta0) res += __bfloat162float(beta) * __bfloat162float(gC[n]);
        gC[n] = __float2bfloat16(res);
    }
}

// TN M==1: C[n] = alpha * sum(A[k] * B[k, n])
template<bool Beta0>
__global__ void bgemv_tn_row_vec8_kernel(
    int N, int K, __nv_bfloat16 alpha,
    const __nv_bfloat16* __restrict__ A, int lda,
    const __nv_bfloat16* __restrict__ B, int ldb,
    __nv_bfloat16 beta, __nv_bfloat16* __restrict__ C,
    int batchCount, long long sA, long long sB, long long sC)
{
    const int b = blockIdx.z; if (b >= batchCount) return;
    const __nv_bfloat16* gA = A + b * sA;
    const __nv_bfloat16* gB = B + b * sB;
    __nv_bfloat16*       gC = C + b * sC;
    
    const int n_start = (blockIdx.x * GEMV_THREADS + threadIdx.x) * 8;
    if (n_start >= N) return;

    __shared__ __nv_bfloat16 smA[GEMV_BK];
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int k0 = 0; k0 < K; k0 += GEMV_BK) {
        int klen = min(GEMV_BK, K - k0);
        if (threadIdx.x < klen) smA[threadIdx.x] = gA[(long long)(k0 + threadIdx.x) * lda];
        __syncthreads();

        for (int ki = 0; ki < klen; ki++) {
            __nv_bfloat16 a_val = smA[ki];
            const float4* b_ptr = (const float4*)&gB[(long long)(k0 + ki) * ldb + n_start];
            float4 b_vec = *b_ptr;
            const __nv_bfloat162* b_bf2 = (const __nv_bfloat162*)&b_vec;
            
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                acc[i*2]   += __bfloat162float(a_val) * __bfloat162float(b_bf2[i].x);
                acc[i*2+1] += __bfloat162float(a_val) * __bfloat162float(b_bf2[i].y);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (n_start + i < N) {
            float res = __bfloat162float(alpha) * acc[i];
            if constexpr (!Beta0) res += __bfloat162float(beta) * __bfloat162float(gC[n_start + i]);
            gC[n_start + i] = __float2bfloat16(res);
        }
    }
}

// NN N==1: C[m] = alpha * sum(A[m, k] * B[k])
template<bool Beta0>
__global__ void bgemv_nn_col_kernel(
    int M, int K, __nv_bfloat16 alpha,
    const __nv_bfloat16* __restrict__ A, int lda,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16 beta, __nv_bfloat16* __restrict__ C,
    int batchCount, long long sA, long long sB, long long sC)
{
    const int b = blockIdx.z; if (b >= batchCount) return;
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int m = blockIdx.x * (GEMV_THREADS / 32) + warp;
    if (m >= M) return;

    const __nv_bfloat16* gA = A + b * sA + (long long)m * lda;
    const __nv_bfloat16* gB = B + b * sB;
    __nv_bfloat16*       gC = C + b * sC;

    float acc = 0.f;
    int k_vec_limit = K / 8;
    const float4* a_vec_ptr = (const float4*)gA;
    const float4* b_vec_ptr = (const float4*)gB;

    for (int k0 = lane; k0 < k_vec_limit; k0 += 32) {
        float4 av = a_vec_ptr[k0];
        float4 bv = b_vec_ptr[k0];
        const __nv_bfloat162* ah2 = (const __nv_bfloat162*)&av;
        const __nv_bfloat162* bh2 = (const __nv_bfloat162*)&bv;
        #pragma unroll
        for(int i=0; i<4; i++) {
            acc += __bfloat162float(ah2[i].x) * __bfloat162float(bh2[i].x);
            acc += __bfloat162float(ah2[i].y) * __bfloat162float(bh2[i].y);
        }
    }
    for (int k = k_vec_limit * 8 + lane; k < K; k += 32) {
        acc += __bfloat162float(gA[k]) * __bfloat162float(gB[k]);
    }

    for (int offset = 16; offset > 0; offset /= 2)
        acc += __shfl_down_sync(0xffffffff, acc, offset);

    if (lane == 0) {
        float res = __bfloat162float(alpha) * acc;
        if constexpr (!Beta0) res += __bfloat162float(beta) * __bfloat162float(gC[m]);
        gC[m] = __float2bfloat16(res);
    }
}

// Wrappers
extern "C" void mycublasBgemv_nn(
    mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha,
    const __nv_bfloat16* A, int lda, long long sA,
    const __nv_bfloat16* B, int ldb, long long sB,
    const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long sC, int batchCount)
{
    cudaStream_t stream = handle ? handle->stream : 0;
    bool b0 = (__bfloat162float(beta) == 0.f);
    if (M == 1) {
        dim3 grid((N / 8 + GEMV_THREADS - 1) / GEMV_THREADS, 1, batchCount);
        if (b0) bgemv_nn_row_vec8_kernel<true> <<<grid, GEMV_THREADS, 0, stream>>>(N, K, alpha, A, B, ldb, beta, C, batchCount, sA, sB, sC);
        else    bgemv_nn_row_vec8_kernel<false><<<grid, GEMV_THREADS, 0, stream>>>(N, K, alpha, A, B, ldb, beta, C, batchCount, sA, sB, sC);
    } else if (N == 1) {
        dim3 grid((M + (GEMV_THREADS/32) - 1) / (GEMV_THREADS/32), 1, batchCount);
        if (b0) bgemv_nn_col_kernel<true> <<<grid, GEMV_THREADS, 0, stream>>>(M, K, alpha, A, lda, B, beta, C, batchCount, sA, sB, sC);
        else    bgemv_nn_col_kernel<false><<<grid, GEMV_THREADS, 0, stream>>>(M, K, alpha, A, lda, B, beta, C, batchCount, sA, sB, sC);
    }
}

extern "C" void mycublasBgemv_nt(
    mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha,
    const __nv_bfloat16* A, int lda, long long sA,
    const __nv_bfloat16* B, int ldb, long long sB,
    const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long sC, int batchCount)
{
    cudaStream_t stream = handle ? handle->stream : 0;
    bool b0 = (__bfloat162float(beta) == 0.f);
    if (M == 1) {
        dim3 grid((N + (GEMV_THREADS/32) - 1) / (GEMV_THREADS/32), 1, batchCount);
        if (b0) bgemv_nt_row_kernel<true> <<<grid, GEMV_THREADS, 0, stream>>>(N, K, alpha, A, B, ldb, beta, C, batchCount, sA, sB, sC);
        else    bgemv_nt_row_kernel<false><<<grid, GEMV_THREADS, 0, stream>>>(N, K, alpha, A, B, ldb, beta, C, batchCount, sA, sB, sC);
    }
}

extern "C" void mycublasBgemv_tn(
    mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha,
    const __nv_bfloat16* A, int lda, long long sA,
    const __nv_bfloat16* B, int ldb, long long sB,
    const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long sC, int batchCount)
{
    cudaStream_t stream = handle ? handle->stream : 0;
    bool b0 = (__bfloat162float(beta) == 0.f);
    if (M == 1) {
        dim3 grid((N / 8 + GEMV_THREADS - 1) / GEMV_THREADS, 1, batchCount);
        if (b0) bgemv_tn_row_vec8_kernel<true> <<<grid, GEMV_THREADS, 0, stream>>>(N, K, alpha, A, lda, B, ldb, beta, C, batchCount, sA, sB, sC);
        else    bgemv_tn_row_vec8_kernel<false><<<grid, GEMV_THREADS, 0, stream>>>(N, K, alpha, A, lda, B, ldb, beta, C, batchCount, sA, sB, sC);
    }
}
