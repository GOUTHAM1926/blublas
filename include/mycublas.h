#ifndef MYCUBLAS_H
#define MYCUBLAS_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Internal context definition
struct mycublasContext {
    int deviceId;
    cudaStream_t stream;
    bool is_sm89;
    int sm_count;
};
typedef struct mycublasContext *mycublasHandle_t;

// Status codes (simplified)
typedef enum {
    MYCUBLAS_STATUS_SUCCESS = 0,
    MYCUBLAS_STATUS_NOT_INITIALIZED = 1,
    MYCUBLAS_STATUS_ALLOC_FAILED = 2,
    MYCUBLAS_STATUS_INVALID_VALUE = 3,
    MYCUBLAS_STATUS_ARCH_MISMATCH = 4,
    MYCUBLAS_STATUS_MAPPING_ERROR = 5,
    MYCUBLAS_STATUS_EXECUTION_FAILED = 6,
    MYCUBLAS_STATUS_INTERNAL_ERROR = 7,
    MYCUBLAS_STATUS_NOT_SUPPORTED = 8,
    MYCUBLAS_STATUS_LICENSE_ERROR = 9
} mycublasStatus_t;

typedef enum {
    MYCUBLAS_OP_N = 0,
    MYCUBLAS_OP_T = 1,
    MYCUBLAS_OP_C = 2
} mycublasOperation_t;

// Management functions
mycublasStatus_t mycublasCreate(mycublasHandle_t *handle);
mycublasStatus_t mycublasDestroy(mycublasHandle_t handle);
mycublasStatus_t mycublasSetStream(mycublasHandle_t handle, cudaStream_t streamId);




















// Level 3 BLAS
void mycublasSgemm(
    mycublasHandle_t handle,
    mycublasOperation_t transa, mycublasOperation_t transb,
    int M, int N, int K, 
    float alpha, 
    const float *d_A, int lda, 
    const float *d_B, int ldb, 
    float beta, 
    float *d_C, int ldc
);

void mycublasSgemm_v2(
    mycublasHandle_t handle, 
    mycublasOperation_t transa, 
    mycublasOperation_t transb, 
    int M, int N, int K, 
    float alpha, const float *A, int lda, 
    const float *B, int ldb, 
    float beta, float *C, int ldc) ;

void mycublasHgemm(
    mycublasHandle_t handle,
    mycublasOperation_t transa, mycublasOperation_t transb,
    int M, int N, int K,
    const __half alpha,
    const __half *d_A, int lda,
    const __half *d_B, int ldb,
    const __half beta,
    __half *d_C, int ldc
);

void mycublasHgemm_v2(
    mycublasHandle_t handle,
    mycublasOperation_t transa, mycublasOperation_t transb,
    int M, int N, int K,
    const __half alpha,
    const __half *d_A, int lda,
    const __half *d_B, int ldb,
    const __half beta,
    __half *d_C, int ldc
);

void mycublasBgemm(
    mycublasHandle_t handle,
    int M, int N, int K,
    const __nv_bfloat16 alpha,
    const __nv_bfloat16 *A, int lda,
    const __nv_bfloat16 *B, int ldb,
    const __nv_bfloat16 beta,
    __nv_bfloat16 *C, int ldc
);

void mycublasBgemm_v2(
    mycublasHandle_t handle,
    int M, int N, int K,
    const __nv_bfloat16 alpha,
    const __nv_bfloat16 *A, int lda,
    const __nv_bfloat16 *B, int ldb,
    const __nv_bfloat16 beta,
    __nv_bfloat16 *C, int ldc
);

void mycublasSgemmStridedBatched(
    mycublasHandle_t handle,
    mycublasOperation_t transa, mycublasOperation_t transb,
    int M, int N, int K,
    const float alpha,
    const float *d_A, int lda, long long int strideA,
    const float *d_B, int ldb, long long int strideB,
    const float beta,
    float *d_C, int ldc, long long int strideC,
    int batchCount
);


    
extern "C" void mycublasSgemmStridedBatched_SM86(
    mycublasHandle_t handle, int M, int N, int K, const float alpha,
    const float* d_A, int lda, long long int strideA,
    const float* d_B, int ldb, long long int strideB,
    const float beta, float* d_C, int ldc, long long int strideC, int batchCount);

extern "C" void mycublasSgemmStridedBatched_splitk_SM86(
    mycublasHandle_t handle, int M, int N, int K, const float alpha,
    const float* d_A, int lda, long long int strideA,
    const float* d_B, int ldb, long long int strideB,
    const float beta, float* d_C, int ldc, long long int strideC, int batchCount);





extern "C" void mycublasSgemmAddmm(
    mycublasHandle_t handle,
    int M, int N, int K, const float alpha,
    const float* d_A, int lda, long long int strideA,
    const float* d_B, int ldb, long long int strideB,
    const float beta, 
    const float* d_bias, int64_t bias_numel,
    float* d_C, int ldc, long long int strideC,
    int batchCount);

extern "C" void mycublasSgemmAddmm_dispatch(
    mycublasHandle_t handle, int M, int N, int K, const float alpha,
    const float* d_A, int lda, long long int strideA,
    const float* d_B, int ldb, long long int strideB,
    const float beta, 
    const float* d_bias, int64_t bias_numel,
    float* d_C, int ldc, long long int strideC, int batchCount);


// GEMV dispatch (M<=16 or N<=16) — NN/NT/TN
extern "C" void mycublasSgemv_nn_SM86(
    mycublasHandle_t handle, int M, int N, int K, float alpha,
    const float* A, int lda, long long sA,
    const float* B, int ldb, long long sB,
    float beta, float* C, int ldc, long long sC, int batchCount);

extern "C" void mycublasSgemv_nt_SM86(
    mycublasHandle_t handle, int M, int N, int K, float alpha,
    const float* A, int lda, long long sA,
    const float* B, int ldb, long long sB,
    float beta, float* C, int ldc, long long sC, int batchCount);

extern "C" void mycublasSgemv_tn_SM86(
    mycublasHandle_t handle, int M, int N, int K, float alpha,
    const float* A, int lda, long long sA,
    const float* B, int ldb, long long sB,
    float beta, float* C, int ldc, long long sC, int batchCount);


extern "C" void mycublasSgemmAddmm_sm89(
    mycublasHandle_t handle,
    int M, int N, int K,
    const float alpha,
    const float* A, int lda, long long int strideA,
    const float* B, int ldb, long long int strideB,
    const float beta,
    const float* bias, int64_t bias_numel,
    float* C, int ldc, long long int strideC,
    int batchCount);

extern "C" void mycublasHgemmAddmm_sm89(
    mycublasHandle_t handle,
    int M, int N, int K,
    const __half alpha,
    const __half* A, int lda, long long int strideA,
    const __half* B, int ldb, long long int strideB,
    const __half beta,
    __half* C, int ldc, long long int strideC,
    const __half* bias, int64_t bias_numel, int batchCount);

extern "C" void mycublasBgemmAddmm_sm89(
    mycublasHandle_t handle,
    int M, int N, int K,
    const __nv_bfloat16 alpha,
    const __nv_bfloat16* A, int lda, long long int strideA,
    const __nv_bfloat16* B, int ldb, long long int strideB,
    const __nv_bfloat16 beta,
    __nv_bfloat16* C, int ldc, long long int strideC,
    const __nv_bfloat16* bias, int64_t bias_numel, int batchCount);

// 64x64 tile dispatch (gx*gy < 8) — NN/NT/TN
extern "C" void mycublasSgemmStridedBatched_nn_tile64_SM86(
    mycublasHandle_t handle, int M, int N, int K, float alpha,
    const float* d_A, int lda, long long strideA,
    const float* d_B, int ldb, long long strideB,
    float beta, float* d_C, int ldc, long long strideC, int batchCount);

extern "C" void mycublasSgemmStridedBatched_nt_tile64_SM86(
    mycublasHandle_t handle, int M, int N, int K, float alpha,
    const float* d_A, int lda, long long strideA,
    const float* d_B, int ldb, long long strideB,
    float beta, float* d_C, int ldc, long long strideC, int batchCount);

extern "C" void mycublasSgemmStridedBatched_tn_tile64_SM86(
    mycublasHandle_t handle, int M, int N, int K, float alpha,
    const float* d_A, int lda, long long strideA,
    const float* d_B, int ldb, long long strideB,
    float beta, float* d_C, int ldc, long long strideC, int batchCount);

extern "C" void mycublasSgemmStridedBatched_nt_SM86(
    mycublasHandle_t handle,
    int M, int N, int K, const float alpha,
    const float* d_A, int lda, long long int strideA,
    const float* d_B, int ldb, long long int strideB,
    const float beta,
    float* d_C, int ldc, long long int strideC, 
    int batchCount);

extern "C" void mycublasSgemmStridedBatched_tn_SM86(
    mycublasHandle_t handle,
    int M, int N, int K, const float alpha,
    const float* d_A, int lda, long long int strideA,
    const float* d_B, int ldb, long long int strideB,
    const float beta,
    float* d_C, int ldc, long long int strideC, 
    int batchCount);

extern "C" void mycublasSgemmStridedBatched_nn_SM89(
    mycublasHandle_t handle, int M, int N, int K, const float alpha,
    const float* d_A, int lda, long long int strideA,
    const float* d_B, int ldb, long long int strideB,
    const float beta, float* d_C, int ldc, long long int strideC, int batchCount);

extern "C" void mycublasSgemmStridedBatched_nt_SM89(
    mycublasHandle_t handle, int M, int N, int K, const float alpha,
    const float* d_A, int lda, long long int strideA,
    const float* d_B, int ldb, long long int strideB,
    const float beta, float* d_C, int ldc, long long int strideC, int batchCount);

extern "C" void mycublasSgemmStridedBatched_tn_SM89(
    mycublasHandle_t handle, int M, int N, int K, const float alpha,
    const float* d_A, int lda, long long int strideA,
    const float* d_B, int ldb, long long int strideB,
    const float beta, float* d_C, int ldc, long long int strideC, int batchCount);


extern "C" void mycublasHgemmStridedBatched(
    mycublasHandle_t handle,
    mycublasOperation_t transa, mycublasOperation_t transb,
    int M, int N, int K,
    const __half alpha,
    const __half *d_A, int lda, long long int strideA,
    const __half *d_B, int ldb, long long int strideB,
    const __half beta,
    __half *d_C, int ldc, long long int strideC,
    int batchCount
);

extern "C" void mycublasHgemmStridedBatched_dispatcher(
    mycublasHandle_t handle,
    mycublasOperation_t transA, mycublasOperation_t transB,
    int M, int N, int K,
    const __half alpha,
    const __half *A, int lda, long long int strideA,
    const __half *B, int ldb, long long int strideB,
    const __half beta,
    __half *C, int ldc, long long int strideC,
    int batchCount);

extern "C" void mycublasBgemmStridedBatched(
    mycublasHandle_t handle,
    mycublasOperation_t transa, mycublasOperation_t transb,
    int M, int N, int K,
    const __nv_bfloat16 alpha,
    const __nv_bfloat16 *A, int lda, long long int strideA,
    const __nv_bfloat16 *B, int ldb, long long int strideB,
    const __nv_bfloat16 beta,
    __nv_bfloat16 *C, int ldc, long long int strideC,
    int batchCount
);

extern "C" void mycublasBgemmStridedBatched_dispatcher(
    mycublasHandle_t handle,
    mycublasOperation_t transA, mycublasOperation_t transB,
    int M, int N, int K,
    const __nv_bfloat16 alpha,
    const __nv_bfloat16 *A, int lda, long long int strideA,
    const __nv_bfloat16 *B, int ldb, long long int strideB,
    const __nv_bfloat16 beta,
    __nv_bfloat16 *C, int ldc, long long int strideC,
    int batchCount);

extern "C" void mycublasSgemmStridedBatched_nt_256x128_16x3(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasSgemmStridedBatched_nt_128x256_16x3(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasSgemmStridedBatched_nt_128x128_16x5(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasSgemmStridedBatched_nt_128x128_32x3(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasSgemmStridedBatched_nt_64x64_16x6(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasSgemmStridedBatched_nt_64x64_32x6(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasSgemmStridedBatched_nt_64x128_32x3(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasSgemmStridedBatched_nt_256x64_16x4(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasSgemmStridedBatched_nt_128x128_16x3(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK);

extern "C" void mycublasSgemmStridedBatched_tn_256x128_16x3(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasSgemmStridedBatched_tn_128x256_16x3(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasSgemmStridedBatched_tn_128x128_16x5(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasSgemmStridedBatched_tn_128x128_32x3(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasSgemmStridedBatched_tn_64x64_16x6(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasSgemmStridedBatched_tn_64x64_32x6(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasSgemmStridedBatched_tn_64x128_32x3(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasSgemmStridedBatched_tn_256x64_16x4(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasSgemmStridedBatched_tn_128x128_16x3(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK);


// sm 86 nn

extern "C" void mycublasSgemmStridedBatched_nn_256x128_16x3(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasSgemmStridedBatched_nn_128x256_16x3(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasSgemmStridedBatched_nn_128x128_16x5(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasSgemmStridedBatched_nn_128x128_32x3(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasSgemmStridedBatched_nn_64x64_16x6(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasSgemmStridedBatched_nn_64x64_32x6(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasSgemmStridedBatched_nn_64x128_32x3(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasSgemmStridedBatched_nn_256x64_16x4(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasSgemmStridedBatched_nn_128x128_16x4(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasSgemmStridedBatched_nn_128x64_16x6(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasSgemmStridedBatched_nn_128x128_16x3(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK);

// sm 86 tn

extern "C" void mycublasSgemmStridedBatched_tn_128x128_16x4(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasSgemmStridedBatched_tn_128x64_16x6(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK);

extern "C" void mycublasSgemmStridedBatched_nt_128x128_16x4(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasSgemmStridedBatched_nt_128x64_16x6(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK);

extern "C" void mycublasSgemmStridedBatched_nt_dispatch(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount);
extern "C" void mycublasSgemmStridedBatched_tn_dispatch(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount);
extern "C" void mycublasSgemmStridedBatched_nn_dispatch(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount);

extern "C" void mycublasSgemmStridedBatched_ampere_128x32_nn(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasSgemmStridedBatched_ampere_32x128_tn(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasSgemmStridedBatched_gemv2T_nn(mycublasHandle_t handle, int N, int K, const float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, const float beta, float* C, int ldc, long long strideC, int batchCount);
extern "C" void mycublasSgemmStridedBatched_gemvNSP_nt(mycublasHandle_t handle, int N, int K, const float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, const float beta, float* C, int ldc, long long strideC, int batchCount, int SplitK);
extern "C" void mycublasSgemmStridedBatched_internal_thin(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, const float beta, float* C, int ldc, long long strideC, int batchCount);


extern "C" void mycublasDgemmStridedBatched(
    mycublasHandle_t handle,
    int M, int N, int K,
    const double alpha,
    const double *A, int lda, long long int strideA,
    const double *B, int ldb, long long int strideB,
    const double beta,
    double *C, int ldc, long long int strideC,
    int batchCount);

extern "C" void mycublasSgemmAddmm_SM86(
    mycublasHandle_t handle, int M, int N, int K, const float alpha,
    const float* d_A, int lda, long long int strideA,
    const float* d_B, int ldb, long long int strideB,
    const float beta, 
    const float* d_bias, int64_t bias_numel,
    float* d_C, int ldc, long long int strideC, int batchCount);


// --- FP16 CUTLASS Variants ---
extern "C" void mycublasHgemmStridedBatched_nn_256x128_32x3(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_nn_128x256_32x3(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_nn_128x128_32x5(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_nn_64x64_32x6(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_nn_256x64_32x4(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_nn_128x64_32x6(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);

extern "C" void mycublasHgemmStridedBatched_nt_256x128_32x3(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_nt_128x256_32x3(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_nt_128x128_32x5(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_nt_64x64_32x6(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_nt_256x64_32x4(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_nt_128x64_32x6(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);

extern "C" void mycublasHgemmStridedBatched_tn_256x128_32x3(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_tn_128x256_32x3(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_tn_128x128_32x5(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_tn_64x64_32x6(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_tn_256x64_32x4(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasHgemmStridedBatched_tn_128x64_32x6(mycublasHandle_t handle, int M, int N, int K, const __half alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half beta, __half* C, int ldc, long long int strideC, int batchCount, int splitK);

// --- BF16 CUTLASS Variants ---
extern "C" void mycublasBgemmStridedBatched_nn_256x128_32x3(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_nn_128x256_32x3(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_nn_128x128_32x5(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_nn_64x64_32x6(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_nn_256x64_32x4(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_nn_128x64_32x6(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);

extern "C" void mycublasBgemmStridedBatched_nt_256x128_32x3(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_nt_128x256_32x3(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_nt_128x128_32x5(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_nt_64x64_32x6(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_nt_256x64_32x4(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_nt_128x64_32x6(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);

extern "C" void mycublasBgemmStridedBatched_tn_256x128_32x3(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_tn_128x256_32x3(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_tn_128x128_32x5(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_tn_64x64_32x6(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_tn_256x64_32x4(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);
extern "C" void mycublasBgemmStridedBatched_tn_128x64_32x6(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK);

// Internal Utility Functions
extern "C" void get_gpu_info(int *sm_ver, int *sm_count);

// ── v47: FP16 peak-performance MMA kernel (128x128x32, 4-warp, 3-stage) ─────
// Handles NN / NT / TN / TT via internal template dispatch.
extern "C" void mycublasHgemmStridedBatched_NN(
    mycublasHandle_t handle,
    mycublasOperation_t transA, mycublasOperation_t transB,
    int M, int N, int K, const __half alpha,
    const __half *A, int lda, long long int strideA,
    const __half *B, int ldb, long long int strideB,
    const __half beta,
    __half *C, int ldc, long long int strideC,
    int batchCount);

// ── v47: BF16 peak-performance MMA kernel (128x128x32, 4-warp, 3-stage) ─────
// Handles NN / NT / TN / TT via internal template dispatch.
extern "C" void mycublasBgemmStridedBatched_NN(
    mycublasHandle_t handle,
    mycublasOperation_t transA, mycublasOperation_t transB,
    int M, int N, int K, const __nv_bfloat16 alpha,
    const __nv_bfloat16 *A, int lda, long long int strideA,
    const __nv_bfloat16 *B, int ldb, long long int strideB,
    const __nv_bfloat16 beta,
    __nv_bfloat16 *C, int ldc, long long int strideC,
    int batchCount);

// =============================================================================
// FP16 (Hgemm/Hgemv) C-Wrappers
// =============================================================================
extern "C" void mycublasHgemv_nn(
    mycublasHandle_t handle, int M, int N, int K, const __half alpha,
    const __half* A, int lda, long long sA,
    const __half* B, int ldb, long long sB,
    const __half beta, __half* C, int ldc, long long sC, int batchCount);

extern "C" void mycublasHgemv_nt(
    mycublasHandle_t handle, int M, int N, int K, const __half alpha,
    const __half* A, int lda, long long sA,
    const __half* B, int ldb, long long sB,
    const __half beta, __half* C, int ldc, long long sC, int batchCount);

extern "C" void mycublasHgemv_tn(
    mycublasHandle_t handle, int M, int N, int K, const __half alpha,
    const __half* A, int lda, long long sA,
    const __half* B, int ldb, long long sB,
    const __half beta, __half* C, int ldc, long long sC, int batchCount);


// =============================================================================
// BF16 (Bgemm/Bgemv) C-Wrappers
// =============================================================================
extern "C" void mycublasBgemv_nn(
    mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha,
    const __nv_bfloat16* A, int lda, long long sA,
    const __nv_bfloat16* B, int ldb, long long sB,
    const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long sC, int batchCount);

extern "C" void mycublasBgemv_nt(
    mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha,
    const __nv_bfloat16* A, int lda, long long sA,
    const __nv_bfloat16* B, int ldb, long long sB,
    const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long sC, int batchCount);

extern "C" void mycublasBgemv_tn(
    mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha,
    const __nv_bfloat16* A, int lda, long long sA,
    const __nv_bfloat16* B, int ldb, long long sB,
    const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long sC, int batchCount);

#ifdef __cplusplus
}
#endif

#endif // MYCUBLAS_H