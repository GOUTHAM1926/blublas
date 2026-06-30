#include "mycublas.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

// Forward declarations of optimized kernels
extern "C" void mycublasHgemmStridedBatched_NN(
    mycublasHandle_t handle, 
    mycublasOperation_t transA, mycublasOperation_t transB,
    int M, int N, int K, const __half alpha,
    const __half *A, int lda, long long int strideA, const __half *B, int ldb, long long int strideB,
    const __half beta, __half *C, int ldc, long long int strideC, int batchCount);

// NN Variants
extern "C" void mycublasHgemmStridedBatched_nn_256x128_32x3(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_nn_128x256_32x3(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_nn_128x128_32x5(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_nn_64x64_32x6(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_nn_128x256_32_5(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_nn_128x256_32_4(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_nn_128x128_32_6(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);

// NT Variants
extern "C" void mycublasHgemmStridedBatched_nt_256x128_32x3(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_nt_128x256_32x3(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_nt_128x128_32x5(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_nt_64x64_32x6(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_nt_256x64_32x4(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_nt_128x64_32x6(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);

// TN Variants
extern "C" void mycublasHgemmStridedBatched_tn_256x128_32x3(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_tn_128x256_32x3(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_tn_128x128_32x5(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_tn_64x64_32x6(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_tn_256x64_32x4(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_tn_128x64_32x6(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);

// SM89 / Advanced Variants
extern "C" void mycublasHgemmStridedBatched_nn_256x128_32_4(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_nn_256x128_32_5(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_nn_128x256_32_5(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_nn_128x256_32_4(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_nn_128x128_32_6(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);

extern "C" void mycublasHgemmStridedBatched_nt_256x128_32_4(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_nt_256x128_32_5(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_nt_128x256_32_5(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_nt_128x256_32_4(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_nt_128x128_32_6(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);

extern "C" void mycublasHgemmStridedBatched_tn_256x128_32_4(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_tn_256x128_32_5(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_tn_128x256_32_5(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_tn_128x256_32_4(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_tn_128x128_32_6(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);

extern "C" void mycublasHgemv_nn(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long sA, const __half* B, int ldb, long long sB, const __half beta, __half* C, int ldc, long long sC, int batchCount);
extern "C" void mycublasHgemv_nt(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long sA, const __half* B, int ldb, long long sB, const __half beta, __half* C, int ldc, long long sC, int batchCount);
extern "C" void mycublasHgemv_tn(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long sA, const __half* B, int ldb, long long sB, const __half beta, __half* C, int ldc, long long sC, int batchCount);

#include "Hgemm_core_template.cuh"

extern "C" void mycublasHgemmAddmm_sm89(
    mycublasHandle_t handle,
    int M, int N, int K,
    const __half alpha,
    const __half* A, int lda, long long int strideA,
    const __half* B, int ldb, long long int strideB,
    const __half beta,
    __half* C, int ldc, long long int strideC,
    const __half* bias, int64_t bias_numel, int batchCount)
{
    cudaStream_t stream = handle ? handle->stream : 0;
    
    // 1. Epilogue pass: Scale + Bias
    if (beta != __float2half(1.0f) || bias != nullptr) {
        dim3 block(32, 32);
        dim3 grid((N + 31) / 32, (M + 31) / 32, batchCount);
        hgemm_scale_template_kernel<HgemmTileConfig<128,128,32,3,128>>
            <<<grid, block, 0, stream>>>(C, beta, M, N, ldc, strideC, batchCount, bias, bias_numel);
    }
    
    // 2. GEMM pass: alpha * A * B + 1.0 * C
    mycublasHgemmStridedBatched_dispatcher(handle, MYCUBLAS_OP_N, MYCUBLAS_OP_N, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, __float2half(1.0f), C, ldc, strideC, batchCount);
}

extern "C" void mycublasHgemmStridedBatched_dispatcher(
    mycublasHandle_t handle,
    mycublasOperation_t transA, mycublasOperation_t transB,
    int M, int N, int K,
    const __half alpha,
    const __half *A, int lda, long long int strideA,
    const __half *B, int ldb, long long int strideB,
    const __half beta,
    __half *C, int ldc, long long int strideC,
    int batchCount)
{
    // Optimization: Use handle-cached hardware flags instead of slow cudaGetDeviceProperties
    bool is_sm89 = handle ? handle->is_sm89 : false;
    int sm_ver = is_sm89 ? 89 : 80; // Simplification for logic below

    // SplitK heuristic
    int splitK = 1;
    if (K >= 4096 && M <= 128 && N <= 128) {
        splitK = 4;
    }

    // 1. SM80+ Specialized Path
    if (sm_ver >= 80) {
        // Layout-specific dispatch
        if (transA == MYCUBLAS_OP_N && transB == MYCUBLAS_OP_N) {
            if (M == 1) {
                mycublasHgemv_nn(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
                return;
            }
            if (is_sm89 && M >= 1024 && N >= 1024) {
                mycublasHgemmStridedBatched_nn_128x128_32_6(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
            } else if (M >= 2048 && N >= 2048) {
                mycublasHgemmStridedBatched_nn_256x128_32_4(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
            } else if (M >= 256 && N >= 128 && M >= N) {
                mycublasHgemmStridedBatched_nn_256x128_32x3(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
            } else if (N >= 256 && M >= 128) {
                mycublasHgemmStridedBatched_nn_128x256_32_4(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
            } else if (M >= 256) {
                mycublasHgemmStridedBatched_nn_256x64_32x4(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
            } else {
                mycublasHgemmStridedBatched_nn_128x128_32_6(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
            }
            return;
        }
        else if (transA == MYCUBLAS_OP_N && transB == MYCUBLAS_OP_T) {
            if (M == 1) {
                mycublasHgemv_nt(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
                return;
            }
            if (is_sm89 && M >= 1024 && N >= 1024) {
                mycublasHgemmStridedBatched_nt_128x128_32_6(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
            } else if (M >= 2048 && N >= 2048) {
                mycublasHgemmStridedBatched_nt_256x128_32_4(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
            } else if (M >= 256 && N >= 128 && M >= N) {
                mycublasHgemmStridedBatched_nt_256x128_32_4(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
            } else if (N >= 256 && M >= 128) {
                mycublasHgemmStridedBatched_nt_128x256_32_4(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
            } else if (M >= 256) {
                mycublasHgemmStridedBatched_nt_256x64_32x4(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
            } else {
                mycublasHgemmStridedBatched_nt_128x128_32_6(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
            }
            return;
        }
        else if (transA == MYCUBLAS_OP_T && transB == MYCUBLAS_OP_N) {
            if (M == 1) {
                mycublasHgemv_tn(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
                return;
            }
            if (is_sm89 && M >= 1024 && N >= 1024) {
                mycublasHgemmStridedBatched_tn_128x128_32_6(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
            } else if (M >= 2048 && N >= 2048) {
                mycublasHgemmStridedBatched_tn_256x128_32_4(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
            } else if (M >= 256 && N >= 128 && M >= N) {
                mycublasHgemmStridedBatched_tn_256x128_32_4(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
            } else if (N >= 256 && M >= 128) {
                mycublasHgemmStridedBatched_tn_128x256_32_4(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
            } else if (M >= 256) {
                mycublasHgemmStridedBatched_tn_256x64_32x4(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
            } else {
                mycublasHgemmStridedBatched_tn_128x128_32_6(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
            }
            return;
        }
    }

    // Default Fallback (could be NN or a basic WMMA kernel)
    mycublasHgemmStridedBatched_NN(handle, transA, transB, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}
