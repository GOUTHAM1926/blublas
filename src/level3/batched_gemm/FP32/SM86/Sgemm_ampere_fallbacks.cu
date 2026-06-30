#include "mycublas.h"
#include <cuda_runtime.h>
#include <unordered_map>
#include "Sgemm_core_template.cuh"

// 9a. ampere_sgemm_128x32_nn
// Small K fallback, standard TF32 MMA, 128x32 tile, NN layout
template <int BM, int BN, int BK, int STAGES, int THREADS, bool IsAligned, int SplitK>
void launch_ampere_nn(
    cudaStream_t stream, int M, int N, int K, const float alpha,
    const float* A, int lda, long long int strideA,
    const float* B, int ldb, long long int strideB,
    const float beta, float* C, int ldc, long long int strideC, 
    int batchCount, int splitK_actual)
{
    using Config = SgemmTileConfig<BM, BN, BK, STAGES, THREADS>;
    static const size_t smem_bytes = STAGES * Config::STAGE_SIZE * sizeof(float);
    
    auto set_attr = [](const void* f, size_t b) {
        static std::unordered_map<const void*, bool> done;
        if (!done[f]) { cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)b); done[f] = true; }
    };
    const void* kernel_func = (const void*)sgemm_backward_template_kernel<Config, IsAligned, SplitK, SgemmLayout::NN>;
    set_attr(kernel_func, smem_bytes);

    const int gx = (N + BN - 1) / BN, gy = (M + BM - 1) / BM;
    dim3 grid(gx, gy, batchCount * splitK_actual);
    
    sgemm_backward_template_kernel<Config, IsAligned, SplitK, SgemmLayout::NN>
        <<<grid, THREADS, smem_bytes, stream>>>(
            M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount
        );
}

extern "C" void mycublasSgemmStridedBatched_ampere_128x32_nn(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK) {
    // 128x32, BK=32, 3-stage, 128 threads
    cudaStream_t stream = handle ? handle->stream : 0;
    const bool aligned = (((size_t)A & 15) == 0) && (((size_t)B & 15) == 0) && ((lda & 3) == 0) && ((ldb & 3) == 0);
    float actual_beta = (splitK > 1) ? 0.0f : beta;

    if (splitK > 1 && beta != 0.0f) {
        dim3 s_grid((N+31)/32, (M+31)/32, batchCount);
        sgemm_scale_template_kernel<SgemmTileConfig<128,32,32,3,128>>
            <<<s_grid, dim3(32,32), 0, stream>>>(C, beta, M, N, ldc, strideC, batchCount);
    }

    if (aligned) {
        if (splitK <= 1) launch_ampere_nn<128, 32, 32, 3, 128, true, 1>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, actual_beta, C, ldc, strideC, batchCount, 1);
        else if (splitK <= 4) launch_ampere_nn<128, 32, 32, 3, 128, true, 4>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, actual_beta, C, ldc, strideC, batchCount, 4);
        else if (splitK <= 8) launch_ampere_nn<128, 32, 32, 3, 128, true, 8>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, actual_beta, C, ldc, strideC, batchCount, 8);
        else launch_ampere_nn<128, 32, 32, 3, 128, true, 16>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, actual_beta, C, ldc, strideC, batchCount, 16);
    } else {
        if (splitK <= 1) launch_ampere_nn<128, 32, 32, 3, 128, false, 1>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, actual_beta, C, ldc, strideC, batchCount, 1);
        else if (splitK <= 4) launch_ampere_nn<128, 32, 32, 3, 128, false, 4>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, actual_beta, C, ldc, strideC, batchCount, 4);
        else if (splitK <= 8) launch_ampere_nn<128, 32, 32, 3, 128, false, 8>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, actual_beta, C, ldc, strideC, batchCount, 8);
        else launch_ampere_nn<128, 32, 32, 3, 128, false, 16>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, actual_beta, C, ldc, strideC, batchCount, 16);
    }
}

// 9b. ampere_sgemm_32x128_tn
// Small K fallback, standard TF32 MMA, 32x128 tile, TN layout
template <int BM, int BN, int BK, int STAGES, int THREADS, bool IsAligned, int SplitK>
void launch_ampere_tn(
    cudaStream_t stream, int M, int N, int K, const float alpha,
    const float* A, int lda, long long int strideA,
    const float* B, int ldb, long long int strideB,
    const float beta, float* C, int ldc, long long int strideC, 
    int batchCount, int splitK_actual)
{
    using Config = SgemmTileConfig<BM, BN, BK, STAGES, THREADS>;
    static const size_t smem_bytes = STAGES * Config::STAGE_SIZE * sizeof(float);
    
    auto set_attr = [](const void* f, size_t b) {
        static std::unordered_map<const void*, bool> done;
        if (!done[f]) { cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)b); done[f] = true; }
    };
    const void* kernel_func = (const void*)sgemm_backward_template_kernel<Config, IsAligned, SplitK, SgemmLayout::TN>;
    set_attr(kernel_func, smem_bytes);

    const int gx = (N + BN - 1) / BN, gy = (M + BM - 1) / BM;
    dim3 grid(gx, gy, batchCount * splitK_actual);
    
    sgemm_backward_template_kernel<Config, IsAligned, SplitK, SgemmLayout::TN>
        <<<grid, THREADS, smem_bytes, stream>>>(
            M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount
        );
}

extern "C" void mycublasSgemmStridedBatched_ampere_32x128_tn(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK) {
    // 32x128, BK=32, 3-stage, 128 threads
    cudaStream_t stream = handle ? handle->stream : 0;
    const bool aligned = (((size_t)A & 15) == 0) && (((size_t)B & 15) == 0) && ((lda & 3) == 0) && ((ldb & 3) == 0);
    float actual_beta = (splitK > 1) ? 0.0f : beta;

    if (splitK > 1 && beta != 0.0f) {
        dim3 s_grid((N+31)/32, (M+31)/32, batchCount);
        sgemm_scale_template_kernel<SgemmTileConfig<32,128,32,3,128>>
            <<<s_grid, dim3(32,32), 0, stream>>>(C, beta, M, N, ldc, strideC, batchCount);
    }

    if (aligned) {
        if (splitK <= 1) launch_ampere_tn<32, 128, 32, 3, 128, true, 1>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, actual_beta, C, ldc, strideC, batchCount, 1);
        else if (splitK <= 4) launch_ampere_tn<32, 128, 32, 3, 128, true, 4>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, actual_beta, C, ldc, strideC, batchCount, 4);
        else if (splitK <= 8) launch_ampere_tn<32, 128, 32, 3, 128, true, 8>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, actual_beta, C, ldc, strideC, batchCount, 8);
        else launch_ampere_tn<32, 128, 32, 3, 128, true, 16>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, actual_beta, C, ldc, strideC, batchCount, 16);
    } else {
        if (splitK <= 1) launch_ampere_tn<32, 128, 32, 3, 128, false, 1>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, actual_beta, C, ldc, strideC, batchCount, 1);
        else if (splitK <= 4) launch_ampere_tn<32, 128, 32, 3, 128, false, 4>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, actual_beta, C, ldc, strideC, batchCount, 4);
        else if (splitK <= 8) launch_ampere_tn<32, 128, 32, 3, 128, false, 8>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, actual_beta, C, ldc, strideC, batchCount, 8);
        else launch_ampere_tn<32, 128, 32, 3, 128, false, 16>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, actual_beta, C, ldc, strideC, batchCount, 16);
    }
}
