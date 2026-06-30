#include "mycublas.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>

// Forward declarations for SM103 Blackwell Ultra dispatcher routing
extern "C" bool mycublasBgemmSM103_dispatch_nn(
    mycublasHandle_t handle,
    int M, int N, int K,
    float alpha,
    const __nv_bfloat16* A, long long strideA,
    const __nv_bfloat16* B, long long strideB,
    float beta,
          __nv_bfloat16* C, long long strideC,
    int batchCount);

extern "C" bool mycublasBgemmSM103_dispatch_nt(
    mycublasHandle_t handle,
    int M, int N, int K,
    float alpha,
    const __nv_bfloat16* A, long long strideA,
    const __nv_bfloat16* B, long long strideB,
    float beta,
          __nv_bfloat16* C, long long strideC,
    int batchCount);

extern "C" bool mycublasBgemmSM103_dispatch_tn(
    mycublasHandle_t handle,
    int M, int N, int K,
    float alpha,
    const __nv_bfloat16* A, long long strideA,
    const __nv_bfloat16* B, long long strideB,
    float beta,
          __nv_bfloat16* C, long long strideC,
    int batchCount);

// Forward declarations of optimized kernels
extern "C" void mycublasBgemmStridedBatched_NN(
    mycublasHandle_t handle,
    mycublasOperation_t transA, mycublasOperation_t transB,
    int M, int N, int K, const __nv_bfloat16 alpha,
    const __nv_bfloat16 *A, int lda, long long int strideA, const __nv_bfloat16 *B, int ldb, long long int strideB,
    const __nv_bfloat16 beta, __nv_bfloat16 *C, int ldc, long long int strideC, int batchCount);

// NN Variants
extern "C" void mycublasBgemmStridedBatched_nn_256x128_32x3(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_nn_128x256_32x3(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_nn_128x128_32x5(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_nn_64x64_32x6(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_nn_128x256_32_5(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_nn_128x256_32_4(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_nn_128x128_32_6(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);

// NT Variants
extern "C" void mycublasBgemmStridedBatched_nt_256x128_32x3(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_nt_128x256_32x3(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_nt_128x128_32x5(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_nt_64x64_32x6(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_nt_256x64_32x4(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_nt_128x64_32x6(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);

// TN Variants
extern "C" void mycublasBgemmStridedBatched_tn_256x128_32x3(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_tn_128x256_32x3(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_tn_128x128_32x5(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_tn_64x64_32x6(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_tn_256x64_32x4(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_tn_128x64_32x6(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);

// SM89 / Advanced Variants
extern "C" void mycublasBgemmStridedBatched_nn_256x128_32_4(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_nn_256x128_32_5(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_nn_128x256_32_5(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_nn_128x256_32_4(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_nn_128x128_32_6(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);

extern "C" void mycublasBgemmStridedBatched_nt_256x128_32_4(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_nt_256x128_32_5(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_nt_128x256_32_5(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_nt_128x256_32_4(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_nt_128x128_32_6(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);

extern "C" void mycublasBgemmStridedBatched_tn_256x128_32_4(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_tn_256x128_32_5(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_tn_128x256_32_5(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_tn_128x256_32_4(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_tn_128x128_32_6(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);

extern "C" void mycublasBgemv_nn(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long sA, const __nv_bfloat16* B, int ldb, long long sB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long sC, int batchCount);
extern "C" void mycublasBgemv_nt(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long sA, const __nv_bfloat16* B, int ldb, long long sB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long sC, int batchCount);
extern "C" void mycublasBgemv_tn(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long sA, const __nv_bfloat16* B, int ldb, long long sB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long sC, int batchCount);

#include "SM89/Bgemm_core_template.cuh"

extern "C" void mycublasBgemmAddmm_sm89(
    mycublasHandle_t handle,
    int M, int N, int K,
    const __nv_bfloat16 alpha,
    const __nv_bfloat16* A, int lda, long long int strideA,
    const __nv_bfloat16* B, int ldb, long long int strideB,
    const __nv_bfloat16 beta,
    __nv_bfloat16* C, int ldc, long long int strideC,
    const __nv_bfloat16* bias, int64_t bias_numel, int batchCount)
{
    cudaStream_t stream = handle ? handle->stream : 0;
    
    // 1. Epilogue pass: Scale + Bias
    if (beta != __float2bfloat16(1.0f) || bias != nullptr) {
        dim3 block(32, 32);
        dim3 grid((N + 31) / 32, (M + 31) / 32, batchCount);
        bgemm_scale_template_kernel<BgemmTileConfig<128,128,32,3,128>>
            <<<grid, block, 0, stream>>>(C, beta, M, N, ldc, strideC, batchCount, bias, bias_numel);
    }
    
    // 2. GEMM pass: alpha * A * B + 1.0 * C
    mycublasBgemmStridedBatched_dispatcher(handle, MYCUBLAS_OP_N, MYCUBLAS_OP_N, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, __float2bfloat16(1.0f), C, ldc, strideC, batchCount);
}

extern "C" void mycublasBgemmStridedBatched_dispatcher(
    mycublasHandle_t handle,
    mycublasOperation_t transA, mycublasOperation_t transB,
    int M, int N, int K,
    const __nv_bfloat16 alpha,
    const __nv_bfloat16 *A, int lda, long long int strideA,
    const __nv_bfloat16 *B, int ldb, long long int strideB,
    const __nv_bfloat16 beta,
    __nv_bfloat16 *C, int ldc, long long int strideC,
    int batchCount)
{
    // 0. SM103 (Blackwell Ultra) Path
    int actual_sm_ver = 80;
    int actual_sm_count = 0;
    get_gpu_info(&actual_sm_ver, &actual_sm_count);
    if (actual_sm_ver >= 103) {
        if (transA == MYCUBLAS_OP_N && transB == MYCUBLAS_OP_N) {
            if (mycublasBgemmSM103_dispatch_nn(handle, M, N, K, __bfloat162float(alpha), A, strideA, B, strideB, __bfloat162float(beta), C, strideC, batchCount)) {
                return;
            }
        } else if (transA == MYCUBLAS_OP_N && transB == MYCUBLAS_OP_T) {
            if (mycublasBgemmSM103_dispatch_nt(handle, M, N, K, __bfloat162float(alpha), A, strideA, B, strideB, __bfloat162float(beta), C, strideC, batchCount)) {
                return;
            }
        } else if (transA == MYCUBLAS_OP_T && transB == MYCUBLAS_OP_N) {
            if (mycublasBgemmSM103_dispatch_tn(handle, M, N, K, __bfloat162float(alpha), A, strideA, B, strideB, __bfloat162float(beta), C, strideC, batchCount)) {
                return;
            }
        }
    }

    // Optimization: Use handle-cached hardware flags instead of slow cudaGetDeviceProperties
    bool is_sm89 = handle ? handle->is_sm89 : false;
    int sm_ver = is_sm89 ? 89 : 80;

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
                mycublasBgemv_nn(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
                return;
            }
            if (is_sm89 && M >= 1024 && N >= 1024) {
                mycublasBgemmStridedBatched_nn_128x128_32_6(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
            } else if (M >= 2048 && N >= 2048) {
                mycublasBgemmStridedBatched_nn_256x128_32_4(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
            } else if (M >= 256 && N >= 128 && M >= N) {
                mycublasBgemmStridedBatched_nn_256x128_32x3(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
            } else if (N >= 256 && M >= 128) {
                mycublasBgemmStridedBatched_nn_128x256_32_4(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
            } else if (M >= 256) {
                mycublasBgemmStridedBatched_nn_256x64_32x4(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
            } else {
                mycublasBgemmStridedBatched_nn_128x128_32_6(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
            }
            return;
        }
        else if (transA == MYCUBLAS_OP_N && transB == MYCUBLAS_OP_T) {
            if (M == 1) {
                mycublasBgemv_nt(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
                return;
            }
            if (is_sm89 && M >= 1024 && N >= 1024) {
                mycublasBgemmStridedBatched_nt_128x128_32_6(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
            } else if (M >= 2048 && N >= 2048) {
                mycublasBgemmStridedBatched_nt_256x128_32_4(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
            } else if (M >= 256 && N >= 128 && M >= N) {
                mycublasBgemmStridedBatched_nt_256x128_32_4(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
            } else if (N >= 256 && M >= 128) {
                mycublasBgemmStridedBatched_nt_128x256_32_4(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
            } else if (M >= 256) {
                mycublasBgemmStridedBatched_nt_256x64_32x4(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
            } else {
                mycublasBgemmStridedBatched_nt_128x128_32_6(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
            }
            return;
        }
        else if (transA == MYCUBLAS_OP_T && transB == MYCUBLAS_OP_N) {
            if (M == 1) {
                mycublasBgemv_tn(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
                return;
            }
            if (is_sm89 && M >= 1024 && N >= 1024) {
                mycublasBgemmStridedBatched_tn_128x128_32_6(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
            } else if (M >= 2048 && N >= 2048) {
                mycublasBgemmStridedBatched_tn_256x128_32_4(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
            } else if (M >= 256 && N >= 128 && M >= N) {
                mycublasBgemmStridedBatched_tn_256x128_32_4(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
            } else if (N >= 256 && M >= 128) {
                mycublasBgemmStridedBatched_tn_128x256_32_4(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
            } else if (M >= 256) {
                mycublasBgemmStridedBatched_tn_256x64_32x4(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
            } else {
                mycublasBgemmStridedBatched_tn_128x128_32_6(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
            }
            return;
        }
    }

    // Default Fallback
    mycublasBgemmStridedBatched_NN(handle, transA, transB, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}
