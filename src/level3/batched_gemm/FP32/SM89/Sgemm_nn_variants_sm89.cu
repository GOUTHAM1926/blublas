#include "mycublas.h"
#include <cuda_runtime.h>
#include <unordered_map>
#include "Sgemm_core_template_sm89.cuh"

// ---------------------------------------------------------------------------
// Helper launcher for SM89 Template (NN Layout)
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int STAGES, int THREADS, bool IsAligned, bool IsSplitK>
void launch_sm89_nn_variant(
    cudaStream_t stream, int M, int N, int K, float alpha,
    const float* A, int lda, long long int strideA,
    const float* B, int ldb, long long int strideB,
    float beta, float* C, int ldc, long long int strideC, 
    int batchCount, int splitK)
{
    using Config = SgemmTileConfigSM89<BM, BN, BK, STAGES, THREADS>;
    static const size_t smem_bytes = STAGES * Config::STAGE_SIZE * sizeof(float);
    
    auto set_attr = [](const void* f, size_t b) {
        static std::unordered_map<const void*, bool> done;
        if (!done[f]) { cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)b); done[f] = true; }
    };
    const void* k = (const void*)sgemm_sm89_kernel<Config, IsAligned, IsSplitK, SgemmLayout::NN>;
    set_attr(k, smem_bytes);

    dim3 block(THREADS);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, batchCount * splitK);
    
    sgemm_sm89_kernel<Config, IsAligned, IsSplitK, SgemmLayout::NN>
        <<<grid, block, smem_bytes, stream>>>(
            M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, nullptr, 0, batchCount, splitK
        );
}

// ---------------------------------------------------------------------------
// Exported Launchers
// ---------------------------------------------------------------------------

extern "C" void launch_sgemm_nn_256x128_sm89_template(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount, int splitK, cudaStream_t stream) {
    const bool aligned = (((size_t)A & 15) == 0) && (((size_t)B & 15) == 0) && ((lda & 3) == 0) && ((ldb & 3) == 0);
    if (aligned) {
        if (splitK > 1) launch_sm89_nn_variant<256, 128, 16, 4, 256, true, true>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
        else launch_sm89_nn_variant<256, 128, 16, 4, 256, true, false>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, 1);
    } else {
        if (splitK > 1) launch_sm89_nn_variant<256, 128, 16, 4, 256, false, true>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
        else launch_sm89_nn_variant<256, 128, 16, 4, 256, false, false>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, 1);
    }
}

extern "C" void launch_sgemm_nn_128x128_sm89_template(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount, int splitK, cudaStream_t stream) {
    const bool aligned = (((size_t)A & 15) == 0) && (((size_t)B & 15) == 0) && ((lda & 3) == 0) && ((ldb & 3) == 0);
    if (aligned) {
        if (splitK > 1) launch_sm89_nn_variant<128, 128, 16, 4, 128, true, true>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
        else launch_sm89_nn_variant<128, 128, 16, 4, 128, true, false>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, 1);
    } else {
        if (splitK > 1) launch_sm89_nn_variant<128, 128, 16, 4, 128, false, true>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
        else launch_sm89_nn_variant<128, 128, 16, 4, 128, false, false>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, 1);
    }
}

extern "C" void launch_sgemm_nn_64x64_sm89_template(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount, int splitK, cudaStream_t stream) {
    const bool aligned = (((size_t)A & 15) == 0) && (((size_t)B & 15) == 0) && ((lda & 3) == 0) && ((ldb & 3) == 0);
    if (aligned) {
        if (splitK > 1) launch_sm89_nn_variant<64, 64, 16, 2, 128, true, true>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
        else launch_sm89_nn_variant<64, 64, 16, 2, 128, true, false>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, 1);
    } else {
        if (splitK > 1) launch_sm89_nn_variant<64, 64, 16, 2, 128, false, true>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
        else launch_sm89_nn_variant<64, 64, 16, 2, 128, false, false>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, 1);
    }
}

extern "C" void launch_sgemm_nn_32x32_sm89_template(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount, int splitK, cudaStream_t stream) {
    const bool aligned = (((size_t)A & 15) == 0) && (((size_t)B & 15) == 0) && ((lda & 3) == 0) && ((ldb & 3) == 0);
    if (aligned) {
        if (splitK > 1) launch_sm89_nn_variant<32, 32, 16, 2, 32, true, true>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
        else launch_sm89_nn_variant<32, 32, 16, 2, 32, true, false>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, 1);
    } else {
        if (splitK > 1) launch_sm89_nn_variant<32, 32, 16, 2, 32, false, true>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
        else launch_sm89_nn_variant<32, 32, 16, 2, 32, false, false>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, 1);
    }
}
