#include "mycublas.h"
#include <cuda_runtime.h>
#include <algorithm>

// Get SM version for architecture-aware dispatching
// Get GPU properties for architecture-aware dispatching
extern "C" void get_gpu_info(int *sm_ver, int *sm_count) {
    static int cached_sm_ver = 0;
    static int cached_sm_count = 0;
    if (cached_sm_ver == 0) {
        int dev;
        cudaGetDevice(&dev);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        cached_sm_ver = prop.major * 10 + prop.minor;
        cached_sm_count = prop.multiProcessorCount;
    }
    if (sm_ver) *sm_ver = cached_sm_ver;
    if (sm_count) *sm_count = cached_sm_count;
}

// Heuristic-based dispatcher for NT layout SGEMM
extern "C" void mycublasSgemmStridedBatched_nt_dispatch(
    mycublasHandle_t handle, int M, int N, int K, const float alpha,
    const float* A, int lda, long long int strideA,
    const float* B, int ldb, long long int strideB,
    const float beta, float* C, int ldc, long long int strideC, 
    int batchCount)
{
    int sm_ver, sm_count;
    get_gpu_info(&sm_ver, &sm_count);

    if (sm_ver >= 89) {
        mycublasSgemmStridedBatched_nt_SM89(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
        return;
    }

    // 2. Prioritize SM86 for standard large shapes (Relaxed to catch Attention kernels)
    if (sm_ver >= 80) {
        if (M >= 64 && N >= 64 && K >= 32) {
            mycublasSgemmStridedBatched_nt_SM86(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
            return;
        }
    }

    // 2. Fallback to specialized CUTLASS variants for non-SM86 shapes
    int gx128 = (N + 127) / 128;
    int gy128 = (M + 127) / 128;
    int total_tiles = gx128 * gy128 * batchCount;

    int splitK = 1;
    // Scaling Factor: Only split K if we can't naturally fill the SMs (sm_count)
    if (total_tiles < sm_count && K >= 2048) {
        if (K >= 32768) splitK = 16;
        else if (K >= 8192) splitK = 8;
        else splitK = 4;
    }

    if (M >= 256 && N >= 128 && ((M/256) * (N/128) * batchCount >= 160)) {
        mycublasSgemmStridedBatched_nt_256x128_16x3(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
    }
    else if (N >= 256 && M >= 128 && ((M/128) * (N/256) * batchCount >= 160)) {
        mycublasSgemmStridedBatched_nt_128x256_16x3(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
    }
    else if (total_tiles < 160 || (M <= 1024 && N <= 1024)) {
        mycublasSgemmStridedBatched_nt_64x64_16x6(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
    }
    else if (M > N * 2 && sm_ver >= 80) {
        mycublasSgemmStridedBatched_nt_256x64_16x4(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
    }
    else if (N > M * 2 && sm_ver >= 80) {
        mycublasSgemmStridedBatched_nt_64x128_32x3(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
    }
    else if (M >= 128 && N >= 128 && K >= 2048) {
        mycublasSgemmStridedBatched_nt_128x128_32x3(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
    }
    else if (M >= 128 && N >= 128 && sm_ver >= 80) {
        mycublasSgemmStridedBatched_nt_128x128_16x4(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
    }
    else if (M >= 128 && N >= 64 && sm_ver >= 80) {
        mycublasSgemmStridedBatched_nt_128x64_16x6(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
    }
    else {
        mycublasSgemmStridedBatched_nt_128x128_16x5(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
    }
}

extern "C" void mycublasSgemmAddmm(
    mycublasHandle_t handle, int M, int N, int K, const float alpha,
    const float* d_A, int lda, long long int strideA,
    const float* d_B, int ldb, long long int strideB,
    const float beta, 
    const float* d_bias, int64_t bias_numel,
    float* d_C, int ldc, long long int strideC, int batchCount)
{
    int sm_ver, sm_count;
    get_gpu_info(&sm_ver, &sm_count);

    if (sm_ver >= 89) {
        mycublasSgemmAddmm_sm89(handle, M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, beta, d_bias, bias_numel, d_C, ldc, strideC, batchCount);
    } else {
        mycublasSgemmAddmm_SM86(handle, M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, beta, d_bias, bias_numel, d_C, ldc, strideC, batchCount);
    }
}

extern "C" void mycublasSgemmAddmm_dispatch(
    mycublasHandle_t handle, int M, int N, int K, const float alpha,
    const float* d_A, int lda, long long int strideA,
    const float* d_B, int ldb, long long int strideB,
    const float beta, 
    const float* d_bias, int64_t bias_numel,
    float* d_C, int ldc, long long int strideC, int batchCount)
{
    mycublasSgemmAddmm(handle, M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, beta, d_bias, bias_numel, d_C, ldc, strideC, batchCount);
}

// Heuristic-based dispatcher for TN layout SGEMM
extern "C" void mycublasSgemmStridedBatched_tn_dispatch(
    mycublasHandle_t handle, int M, int N, int K, const float alpha,
    const float* A, int lda, long long int strideA,
    const float* B, int ldb, long long int strideB,
    const float beta, float* C, int ldc, long long int strideC, 
    int batchCount)
{
    int sm_ver, sm_count;
    get_gpu_info(&sm_ver, &sm_count);

    if (sm_ver >= 89) {
        mycublasSgemmStridedBatched_tn_SM89(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
        return;
    }

    // 1. Prioritize SM86 for standard large shapes
    if (sm_ver >= 80) {
        if (M >= 64 && N >= 64 && K >= 32) {
            mycublasSgemmStridedBatched_tn_SM86(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
            return;
        }
    }

    int gx128 = (N + 127) / 128;
    int gy128 = (M + 127) / 128;
    int total_tiles = gx128 * gy128 * batchCount;

    int splitK = 1;
    if (total_tiles < sm_count && K >= 2048) {
        if (K >= 32768) splitK = 16;
        else if (K >= 8192) splitK = 8;
        else splitK = 4;
    }

    if (M >= 256 && N >= 128 && ((M/256) * (N/128) * batchCount >= 160)) {
        mycublasSgemmStridedBatched_tn_256x128_16x3(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
    }
    else if (N >= 256 && M >= 128 && ((M/128) * (N/256) * batchCount >= 160)) {
        mycublasSgemmStridedBatched_tn_128x256_16x3(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
    }
    else if (total_tiles < 160 || (M <= 1024 && N <= 1024)) {
        mycublasSgemmStridedBatched_tn_64x64_16x6(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
    }
    else if (M > N * 2 && sm_ver >= 80) {
        mycublasSgemmStridedBatched_tn_256x64_16x4(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
    }
    else if (N > M * 2 && sm_ver >= 80) {
        mycublasSgemmStridedBatched_tn_64x128_32x3(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
    }
    else if (M >= 128 && N >= 128 && K >= 2048) {
        mycublasSgemmStridedBatched_tn_128x128_32x3(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
    }
    else if (M >= 128 && N >= 128 && sm_ver >= 80) {
        mycublasSgemmStridedBatched_tn_128x128_16x4(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
    }
    else if (M >= 128 && N >= 64 && sm_ver >= 80) {
        mycublasSgemmStridedBatched_tn_128x64_16x6(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
    }
    else {
        mycublasSgemmStridedBatched_tn_128x128_16x5(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
    }
}

// Heuristic-based dispatcher for NN layout SGEMM (Forward)
extern "C" void mycublasSgemmStridedBatched_nn_dispatch(
    mycublasHandle_t handle, int M, int N, int K, const float alpha,
    const float* A, int lda, long long int strideA,
    const float* B, int ldb, long long int strideB,
    const float beta, float* C, int ldc, long long int strideC, 
    int batchCount)
{
    int sm_ver, sm_count;
    get_gpu_info(&sm_ver, &sm_count);

    if (sm_ver >= 89) {
        mycublasSgemmStridedBatched_nn_SM89(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
        return;
    }

    // 1. Prioritize SM86 for standard large shapes
    if (sm_ver >= 80) {
        if (M >= 64 && N >= 64 && K >= 32) {
            mycublasSgemmStridedBatched_SM86(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
            return;
        }
    }

    int gx128 = (N + 127) / 128;
    int gy128 = (M + 127) / 128;
    int total_tiles = gx128 * gy128 * batchCount;
    
    int splitK = 1;
    if (total_tiles < sm_count && K >= 2048) {
        if (K >= 32768) splitK = 16;
        else if (K >= 8192) splitK = 8;
        else splitK = 4;
    }

    if (M >= 256 && N >= 128 && ((M/256) * (N/128) * batchCount >= 160)) {
        mycublasSgemmStridedBatched_nn_256x128_16x3(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
    }
    else if (N >= 256 && M >= 128 && ((M/128) * (N/256) * batchCount >= 160)) {
        mycublasSgemmStridedBatched_nn_128x256_16x3(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
    }
    else if (total_tiles < 160 || (M <= 1024 && N <= 1024)) {
        mycublasSgemmStridedBatched_nn_64x64_16x6(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
    }
    else if (M > N * 2 && sm_ver >= 80) {
        mycublasSgemmStridedBatched_nn_256x64_16x4(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
    }
    else if (N > M * 2 && sm_ver >= 80) {
        mycublasSgemmStridedBatched_nn_64x128_32x3(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
    }
    else if (M >= 128 && N >= 128 && K >= 2048) {
        mycublasSgemmStridedBatched_nn_128x128_32x3(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
    }
    else if (M >= 128 && N >= 128 && sm_ver >= 80) {
        mycublasSgemmStridedBatched_nn_128x128_16x4(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
    }
    else if (M >= 128 && N >= 64 && sm_ver >= 80) {
        mycublasSgemmStridedBatched_nn_128x64_16x6(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
    }
    else {
        mycublasSgemmStridedBatched_nn_128x128_16x5(handle, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
    }
}