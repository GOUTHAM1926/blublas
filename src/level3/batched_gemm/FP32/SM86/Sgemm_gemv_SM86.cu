#include "mycublas.h"
#include <cuda_runtime.h>

// ============================================================
// GEMV Dispatch Kernels V18
//
// Handles the "thin" regime: M<=16 or N<=16.
// A 128x128 tile wastes >87% of TensorCore work in these cases.
//
// Strategy:
//   256 threads per block, each thread owns one output column n.
//   A values are loaded cooperatively into smem and broadcast.
//   B reads are coalesced across threads (consecutive n).
//
// Three op types: NN, NT, TN.
// Two shapes per op: row-thin (M==1 or M<=BM_THIN), col-thin (N==1).
// ============================================================

#define GEMV_THREADS  256
#define GEMV_BK       256   // K-tile size loaded into smem per outer loop
#define GEMV_BM       16    // M rows processed per block in thin kernels

// ============================================================
// Helper: write output with or without beta scaling
// ============================================================
__device__ __forceinline__ void store_out(float* dst, float acc, float alpha, float beta, bool beta0) {
    *dst = beta0 ? alpha * acc : alpha * acc + beta * (*dst);
}

// ============================================================
// NN  M==1 : C[n] = alpha * A[k] * B[k,n] summed over k
// A is [1,K] (treated as length-K vector), B is [K,N] row-major
// ============================================================
template<bool Beta0>
__global__ void __launch_bounds__(GEMV_THREADS)
sgemv_nn_row_kernel(
    int N, int K, float alpha,
    const float* __restrict__ A,
    const float* __restrict__ B, int ldb,
    float beta, float* __restrict__ C,
    int batchCount, long long sA, long long sB, long long sC)
{
    const int b = blockIdx.z; if (b >= batchCount) return;
    const float* gA = A + b * sA;
    const float* gB = B + b * sB;
    float*       gC = C + b * sC;
    const int n = blockIdx.x * GEMV_THREADS + threadIdx.x;

    __shared__ float smA[GEMV_BK];
    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += GEMV_BK) {
        int klen = min(GEMV_BK, K - k0);
        if (threadIdx.x < klen) smA[threadIdx.x] = gA[k0 + threadIdx.x];
        __syncthreads();
        if (n < N)
            for (int ki = 0; ki < klen; ki++)
                acc += smA[ki] * gB[(long long)(k0 + ki) * ldb + n];
        __syncthreads();
    }
    if (n < N) {
        if constexpr (Beta0) gC[n] = alpha * acc;
        else                  gC[n] = alpha * acc + beta * gC[n];
    }
}

// ============================================================
// NN  N==1 : C[m] = alpha * A[m,k] * B[k] summed over k
// A is [M,K] row-major, B is [K,1] vector
// ============================================================
template<bool Beta0>
__global__ void __launch_bounds__(GEMV_THREADS)
sgemv_nn_col_kernel(
    int M, int K, float alpha,
    const float* __restrict__ A, int lda,
    const float* __restrict__ B,
    float beta, float* __restrict__ C,
    int batchCount, long long sA, long long sB, long long sC)
{
    const int b = blockIdx.z; if (b >= batchCount) return;
    const float* gA = A + b * sA;
    const float* gB = B + b * sB;
    float*       gC = C + b * sC;
    const int m = blockIdx.x * GEMV_THREADS + threadIdx.x;

    __shared__ float smB[GEMV_BK];
    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += GEMV_BK) {
        int klen = min(GEMV_BK, K - k0);
        if (threadIdx.x < klen) smB[threadIdx.x] = gB[k0 + threadIdx.x];
        __syncthreads();
        if (m < M)
            for (int ki = 0; ki < klen; ki++)
                acc += gA[(long long)m * lda + k0 + ki] * smB[ki];
        __syncthreads();
    }
    if (m < M) {
        if constexpr (Beta0) gC[m] = alpha * acc;
        else                  gC[m] = alpha * acc + beta * gC[m];
    }
}

// ============================================================
// NN  thin (M in [2..16] or N in [2..16], neither ==1)
// Each block: GEMV_THREADS cols x GEMV_BM rows
// ============================================================
template<bool Beta0>
__global__ void __launch_bounds__(GEMV_THREADS)
sgemv_nn_thin_kernel(
    int M, int N, int K, float alpha,
    const float* __restrict__ A, int lda,
    const float* __restrict__ B, int ldb,
    float beta, float* __restrict__ C, int ldc,
    int batchCount, long long sA, long long sB, long long sC)
{
    const int b = blockIdx.z; if (b >= batchCount) return;
    const float* gA = A + b * sA;
    const float* gB = B + b * sB;
    float*       gC = C + b * sC;

    const int n  = blockIdx.x * GEMV_THREADS + threadIdx.x;
    const int m0 = blockIdx.y * GEMV_BM;

    // smA[mi][ki]: A tile, GEMV_BM rows x GEMV_BK cols
    __shared__ float smA[GEMV_BM][GEMV_BK + 1]; // +1 avoids bank conflicts

    float acc[GEMV_BM] = {};

    for (int k0 = 0; k0 < K; k0 += GEMV_BK) {
        int klen = min(GEMV_BK, K - k0);
        // Load smA: GEMV_BM * GEMV_BK = 16*256 = 4096 floats, 256 threads → 16 each
        for (int i = 0; i < GEMV_BM * GEMV_BK / GEMV_THREADS; i++) {
            int idx = i * GEMV_THREADS + threadIdx.x;
            int mi  = idx / GEMV_BK, ki = idx % GEMV_BK;
            int gm  = m0 + mi, gk = k0 + ki;
            smA[mi][ki] = (gm < M && gk < K) ? gA[(long long)gm * lda + gk] : 0.f;
        }
        __syncthreads();

        // Each thread reads its own B column directly — avoids diagonal smB bug
        if (n < N) {
            for (int ki = 0; ki < klen; ki++) {
                float bval = gB[(long long)(k0 + ki) * ldb + n];
                for (int mi = 0; mi < GEMV_BM; mi++)
                    acc[mi] += smA[mi][ki] * bval;
            }
        }
        __syncthreads();
    }

    if (n < N) {
        for (int mi = 0; mi < GEMV_BM; mi++) {
            int gm = m0 + mi;
            if (gm < M) {
                float* dst = &gC[(long long)gm * ldc + n];
                if constexpr (Beta0) *dst = alpha * acc[mi];
                else                  *dst = alpha * acc[mi] + beta * (*dst);
            }
        }
    }
}

// ============================================================
// NT  M==1 : C[n] = alpha * A[k] * B[n,k] summed over k   (B is [N,K])
// ============================================================
template<bool Beta0>
__global__ void __launch_bounds__(GEMV_THREADS)
sgemv_nt_row_kernel(
    int N, int K, float alpha,
    const float* __restrict__ A,
    const float* __restrict__ B, int ldb,
    float beta, float* __restrict__ C,
    int batchCount, long long sA, long long sB, long long sC)
{
    const int b = blockIdx.z; if (b >= batchCount) return;
    const float* gA = A + b * sA;
    const float* gB = B + b * sB;
    float*       gC = C + b * sC;
    const int n = blockIdx.x * GEMV_THREADS + threadIdx.x;

    __shared__ float smA[GEMV_BK];
    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += GEMV_BK) {
        int klen = min(GEMV_BK, K - k0);
        if (threadIdx.x < klen) smA[threadIdx.x] = gA[k0 + threadIdx.x];
        __syncthreads();
        if (n < N)
            for (int ki = 0; ki < klen; ki++)
                acc += smA[ki] * gB[(long long)n * ldb + k0 + ki];
        __syncthreads();
    }
    if (n < N) {
        if constexpr (Beta0) gC[n] = alpha * acc;
        else                  gC[n] = alpha * acc + beta * gC[n];
    }
}

// ============================================================
// NT  N==1 : C[m] = alpha * A[m,k] * B[0,k] summed over k  (B is [1,K])
// ============================================================
template<bool Beta0>
__global__ void __launch_bounds__(GEMV_THREADS)
sgemv_nt_col_kernel(
    int M, int K, float alpha,
    const float* __restrict__ A, int lda,
    const float* __restrict__ B,
    float beta, float* __restrict__ C,
    int batchCount, long long sA, long long sB, long long sC)
{
    const int b = blockIdx.z; if (b >= batchCount) return;
    const float* gA = A + b * sA;
    const float* gB = B + b * sB;
    float*       gC = C + b * sC;
    const int m = blockIdx.x * GEMV_THREADS + threadIdx.x;

    __shared__ float smB[GEMV_BK];
    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += GEMV_BK) {
        int klen = min(GEMV_BK, K - k0);
        if (threadIdx.x < klen) smB[threadIdx.x] = gB[k0 + threadIdx.x];
        __syncthreads();
        if (m < M)
            for (int ki = 0; ki < klen; ki++)
                acc += gA[(long long)m * lda + k0 + ki] * smB[ki];
        __syncthreads();
    }
    if (m < M) {
        if constexpr (Beta0) gC[m] = alpha * acc;
        else                  gC[m] = alpha * acc + beta * gC[m];
    }
}

// ============================================================
// NT  thin: C[M,N] = alpha * A[M,K] * B[N,K]^T + beta*C
// ============================================================
template<bool Beta0>
__global__ void __launch_bounds__(GEMV_THREADS)
sgemv_nt_thin_kernel(
    int M, int N, int K, float alpha,
    const float* __restrict__ A, int lda,
    const float* __restrict__ B, int ldb,
    float beta, float* __restrict__ C, int ldc,
    int batchCount, long long sA, long long sB, long long sC)
{
    const int b = blockIdx.z; if (b >= batchCount) return;
    const float* gA = A + b * sA;
    const float* gB = B + b * sB;
    float*       gC = C + b * sC;

    const int n  = blockIdx.x * GEMV_THREADS + threadIdx.x;
    const int m0 = blockIdx.y * GEMV_BM;

    __shared__ float smA[GEMV_BM][GEMV_BK + 1];
    float acc[GEMV_BM] = {};

    for (int k0 = 0; k0 < K; k0 += GEMV_BK) {
        int klen = min(GEMV_BK, K - k0);
        for (int i = 0; i < GEMV_BM * GEMV_BK / GEMV_THREADS; i++) {
            int idx = i * GEMV_THREADS + threadIdx.x;
            int mi = idx / GEMV_BK, ki = idx % GEMV_BK;
            int gm = m0 + mi, gk = k0 + ki;
            smA[mi][ki] = (gm < M && gk < K) ? gA[(long long)gm * lda + gk] : 0.f;
        }
        __syncthreads();
        // Each thread reads its own B row directly (B is N×K for NT)
        if (n < N) {
            for (int ki = 0; ki < klen; ki++) {
                float bval = gB[(long long)n * ldb + k0 + ki];
                for (int mi = 0; mi < GEMV_BM; mi++)
                    acc[mi] += smA[mi][ki] * bval;
            }
        }
        __syncthreads();
    }

    if (n < N) {
        for (int mi = 0; mi < GEMV_BM; mi++) {
            int gm = m0 + mi;
            if (gm < M) {
                float* dst = &gC[(long long)gm * ldc + n];
                if constexpr (Beta0) *dst = alpha * acc[mi];
                else                  *dst = alpha * acc[mi] + beta * (*dst);
            }
        }
    }
}

// ============================================================
// TN  M==1 : C[n] = alpha * A^T[0,k] * B[k,n] summed over k
// A is stored [M,K] row-major with M=1, but the TN convention is
// A_stored[K,M] where A^T[m,k]=A_stored[k*lda+m].
// When M=1: A^T[0,k] = A_stored[k*lda+0] = A_stored[k*lda]
// ============================================================
template<bool Beta0>
__global__ void __launch_bounds__(GEMV_THREADS)
sgemv_tn_row_kernel(
    int N, int K, float alpha,
    const float* __restrict__ A, int lda,  // A_stored is [K,M] so A^T[0,k]=A[k*lda]
    const float* __restrict__ B, int ldb,
    float beta, float* __restrict__ C,
    int batchCount, long long sA, long long sB, long long sC)
{
    const int b = blockIdx.z; if (b >= batchCount) return;
    const float* gA = A + b * sA;
    const float* gB = B + b * sB;
    float*       gC = C + b * sC;
    const int n = blockIdx.x * GEMV_THREADS + threadIdx.x;

    __shared__ float smA[GEMV_BK];
    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += GEMV_BK) {
        int klen = min(GEMV_BK, K - k0);
        if (threadIdx.x < klen)
            smA[threadIdx.x] = gA[(long long)(k0 + threadIdx.x) * lda];
        __syncthreads();
        if (n < N)
            for (int ki = 0; ki < klen; ki++)
                acc += smA[ki] * gB[(long long)(k0 + ki) * ldb + n];
        __syncthreads();
    }
    if (n < N) {
        if constexpr (Beta0) gC[n] = alpha * acc;
        else                  gC[n] = alpha * acc + beta * gC[n];
    }
}

// ============================================================
// TN  N==1 : C[m] = alpha * A^T[m,k] * B[k] summed over k
// A^T[m,k] = A_stored[k*lda + m]
// ============================================================
template<bool Beta0>
__global__ void __launch_bounds__(GEMV_THREADS)
sgemv_tn_col_kernel(
    int M, int K, float alpha,
    const float* __restrict__ A, int lda,
    const float* __restrict__ B,
    float beta, float* __restrict__ C,
    int batchCount, long long sA, long long sB, long long sC)
{
    const int b = blockIdx.z; if (b >= batchCount) return;
    const float* gA = A + b * sA;
    const float* gB = B + b * sB;
    float*       gC = C + b * sC;
    const int m = blockIdx.x * GEMV_THREADS + threadIdx.x;

    __shared__ float smB[GEMV_BK];
    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += GEMV_BK) {
        int klen = min(GEMV_BK, K - k0);
        if (threadIdx.x < klen) smB[threadIdx.x] = gB[k0 + threadIdx.x];
        __syncthreads();
        if (m < M)
            for (int ki = 0; ki < klen; ki++)
                acc += gA[(long long)(k0 + ki) * lda + m] * smB[ki];
        __syncthreads();
    }
    if (m < M) {
        if constexpr (Beta0) gC[m] = alpha * acc;
        else                  gC[m] = alpha * acc + beta * gC[m];
    }
}

// ============================================================
// TN  thin: C[M,N] = alpha * A^T[M,K] * B[K,N] + beta*C
// A^T[m,k] = A_stored[k*lda + m]
// ============================================================
template<bool Beta0>
__global__ void __launch_bounds__(GEMV_THREADS)
sgemv_tn_thin_kernel(
    int M, int N, int K, float alpha,
    const float* __restrict__ A, int lda,
    const float* __restrict__ B, int ldb,
    float beta, float* __restrict__ C, int ldc,
    int batchCount, long long sA, long long sB, long long sC)
{
    const int b = blockIdx.z; if (b >= batchCount) return;
    const float* gA = A + b * sA;
    const float* gB = B + b * sB;
    float*       gC = C + b * sC;

    const int n  = blockIdx.x * GEMV_THREADS + threadIdx.x;
    const int m0 = blockIdx.y * GEMV_BM;

    // smA[ki][mi]: A^T tile stored as [K,M] in smem
    __shared__ float smA[GEMV_BK][GEMV_BM + 1];
    float acc[GEMV_BM] = {};

    for (int k0 = 0; k0 < K; k0 += GEMV_BK) {
        int klen = min(GEMV_BK, K - k0);
        // Load smA: A^T[m][k] = A_stored[k*lda + m]
        for (int i = 0; i < GEMV_BM * GEMV_BK / GEMV_THREADS; i++) {
            int idx = i * GEMV_THREADS + threadIdx.x;
            int ki  = idx / GEMV_BM, mi = idx % GEMV_BM;
            int gk  = k0 + ki, gm = m0 + mi;
            smA[ki][mi] = (gk < K && gm < M) ? gA[(long long)gk * lda + gm] : 0.f;
        }
        __syncthreads();
        // Each thread reads its own B column directly (B is K×N for TN)
        if (n < N) {
            for (int ki = 0; ki < klen; ki++) {
                float bval = gB[(long long)(k0 + ki) * ldb + n];
                for (int mi = 0; mi < GEMV_BM; mi++)
                    acc[mi] += smA[ki][mi] * bval;
            }
        }
        __syncthreads();
    }

    if (n < N) {
        for (int mi = 0; mi < GEMV_BM; mi++) {
            int gm = m0 + mi;
            if (gm < M) {
                float* dst = &gC[(long long)gm * ldc + n];
                if constexpr (Beta0) *dst = alpha * acc[mi];
                else                  *dst = alpha * acc[mi] + beta * (*dst);
            }
        }
    }
}

// ============================================================
// Public dispatch functions — called from NN/NT/TN dispatch files
// ============================================================

extern "C" void mycublasSgemv_nn_SM86(
    mycublasHandle_t handle, int M, int N, int K, float alpha,
    const float* A, int lda, long long sA,
    const float* B, int ldb, long long sB,
    float beta, float* C, int ldc, long long sC, int batchCount)
{
    cudaStream_t stream = handle ? handle->stream : 0;
    bool b0 = (beta == 0.f);
    if (M == 1) {
        dim3 grid((N + GEMV_THREADS - 1) / GEMV_THREADS, 1, batchCount);
        if (b0) sgemv_nn_row_kernel<true> <<<grid, GEMV_THREADS, 0, stream>>>(N,K,alpha,A,B,ldb,beta,C,batchCount,sA,sB,sC);
        else    sgemv_nn_row_kernel<false><<<grid, GEMV_THREADS, 0, stream>>>(N,K,alpha,A,B,ldb,beta,C,batchCount,sA,sB,sC);
    } else if (N == 1) {
        dim3 grid((M + GEMV_THREADS - 1) / GEMV_THREADS, 1, batchCount);
        if (b0) sgemv_nn_col_kernel<true> <<<grid, GEMV_THREADS, 0, stream>>>(M,K,alpha,A,lda,B,beta,C,batchCount,sA,sB,sC);
        else    sgemv_nn_col_kernel<false><<<grid, GEMV_THREADS, 0, stream>>>(M,K,alpha,A,lda,B,beta,C,batchCount,sA,sB,sC);
    } else {
        dim3 grid((N + GEMV_THREADS - 1) / GEMV_THREADS, (M + GEMV_BM - 1) / GEMV_BM, batchCount);
        if (b0) sgemv_nn_thin_kernel<true> <<<grid, GEMV_THREADS, 0, stream>>>(M,N,K,alpha,A,lda,B,ldb,beta,C,ldc,batchCount,sA,sB,sC);
        else    sgemv_nn_thin_kernel<false><<<grid, GEMV_THREADS, 0, stream>>>(M,N,K,alpha,A,lda,B,ldb,beta,C,ldc,batchCount,sA,sB,sC);
    }
}

extern "C" void mycublasSgemv_nt_SM86(
    mycublasHandle_t handle, int M, int N, int K, float alpha,
    const float* A, int lda, long long sA,
    const float* B, int ldb, long long sB,
    float beta, float* C, int ldc, long long sC, int batchCount)
{
    cudaStream_t stream = handle ? handle->stream : 0;
    bool b0 = (beta == 0.f);
    if (M == 1) {
        dim3 grid((N + GEMV_THREADS - 1) / GEMV_THREADS, 1, batchCount);
        if (b0) sgemv_nt_row_kernel<true> <<<grid, GEMV_THREADS, 0, stream>>>(N,K,alpha,A,B,ldb,beta,C,batchCount,sA,sB,sC);
        else    sgemv_nt_row_kernel<false><<<grid, GEMV_THREADS, 0, stream>>>(N,K,alpha,A,B,ldb,beta,C,batchCount,sA,sB,sC);
    } else if (N == 1) {
        dim3 grid((M + GEMV_THREADS - 1) / GEMV_THREADS, 1, batchCount);
        if (b0) sgemv_nt_col_kernel<true> <<<grid, GEMV_THREADS, 0, stream>>>(M,K,alpha,A,lda,B,beta,C,batchCount,sA,sB,sC);
        else    sgemv_nt_col_kernel<false><<<grid, GEMV_THREADS, 0, stream>>>(M,K,alpha,A,lda,B,beta,C,batchCount,sA,sB,sC);
    } else {
        dim3 grid((N + GEMV_THREADS - 1) / GEMV_THREADS, (M + GEMV_BM - 1) / GEMV_BM, batchCount);
        if (b0) sgemv_nt_thin_kernel<true> <<<grid, GEMV_THREADS, 0, stream>>>(M,N,K,alpha,A,lda,B,ldb,beta,C,ldc,batchCount,sA,sB,sC);
        else    sgemv_nt_thin_kernel<false><<<grid, GEMV_THREADS, 0, stream>>>(M,N,K,alpha,A,lda,B,ldb,beta,C,ldc,batchCount,sA,sB,sC);
    }
}

extern "C" void mycublasSgemv_tn_SM86(
    mycublasHandle_t handle, int M, int N, int K, float alpha,
    const float* A, int lda, long long sA,
    const float* B, int ldb, long long sB,
    float beta, float* C, int ldc, long long sC, int batchCount)
{
    cudaStream_t stream = handle ? handle->stream : 0;
    bool b0 = (beta == 0.f);
    if (M == 1) {
        dim3 grid((N + GEMV_THREADS - 1) / GEMV_THREADS, 1, batchCount);
        if (b0) sgemv_tn_row_kernel<true> <<<grid, GEMV_THREADS, 0, stream>>>(N,K,alpha,A,lda,B,ldb,beta,C,batchCount,sA,sB,sC);
        else    sgemv_tn_row_kernel<false><<<grid, GEMV_THREADS, 0, stream>>>(N,K,alpha,A,lda,B,ldb,beta,C,batchCount,sA,sB,sC);
    } else if (N == 1) {
        dim3 grid((M + GEMV_THREADS - 1) / GEMV_THREADS, 1, batchCount);
        if (b0) sgemv_tn_col_kernel<true> <<<grid, GEMV_THREADS, 0, stream>>>(M,K,alpha,A,lda,B,beta,C,batchCount,sA,sB,sC);
        else    sgemv_tn_col_kernel<false><<<grid, GEMV_THREADS, 0, stream>>>(M,K,alpha,A,lda,B,beta,C,batchCount,sA,sB,sC);
    } else {
        dim3 grid((N + GEMV_THREADS - 1) / GEMV_THREADS, (M + GEMV_BM - 1) / GEMV_BM, batchCount);
        if (b0) sgemv_tn_thin_kernel<true> <<<grid, GEMV_THREADS, 0, stream>>>(M,N,K,alpha,A,lda,B,ldb,beta,C,ldc,batchCount,sA,sB,sC);
        else    sgemv_tn_thin_kernel<false><<<grid, GEMV_THREADS, 0, stream>>>(M,N,K,alpha,A,lda,B,ldb,beta,C,ldc,batchCount,sA,sB,sC);
    }
}
