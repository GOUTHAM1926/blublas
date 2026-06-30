#include "mycublas.h"
#include <cuda_runtime.h>
#include <unordered_map>
#include "Sgemm_core_template.cuh"

// Helper to launch
template <int BM, int BN, int BK, int STAGES, int THREADS, bool IsAligned, int SplitK>
void launch_sgemm_nt_variant(
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
    const void* kernel_func = (const void*)sgemm_backward_template_kernel<Config, IsAligned, SplitK, SgemmLayout::NT>;
    set_attr(kernel_func, smem_bytes);

    const int gx = (N + BN - 1) / BN, gy = (M + BM - 1) / BM;
    dim3 grid(gx, gy, batchCount * splitK_actual);
    
    sgemm_backward_template_kernel<Config, IsAligned, SplitK, SgemmLayout::NT>
        <<<grid, THREADS, smem_bytes, stream>>>(
            M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount
        );
}

template <int BM, int BN, int BK, int STAGES, int THREADS>
void dispatch_sgemm_nt(
    cudaStream_t stream, int M, int N, int K, const float alpha,
    const float* A, int lda, long long int strideA,
    const float* B, int ldb, long long int strideB,
    const float beta, float* C, int ldc, long long int strideC, 
    int batchCount, int splitK)
{
    const bool aligned = (((size_t)A & 15) == 0) && ((lda & 3) == 0) && 
                         (((size_t)B & 15) == 0) && ((ldb & 3) == 0);
                         
    if (splitK > 1) {
        dim3 s_grid((N+31)/32, (M+31)/32, batchCount);
        sgemm_scale_template_kernel<SgemmTileConfig<BM,BN,BK,STAGES,THREADS>>
            <<<s_grid, dim3(32,32), 0, stream>>>(C, beta, M, N, ldc, strideC, batchCount);
    }
    float actual_beta = (splitK > 1) ? 0.0f : beta;

    if (aligned) {
        if (splitK <= 1) launch_sgemm_nt_variant<BM, BN, BK, STAGES, THREADS, true, 1>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, actual_beta, C, ldc, strideC, batchCount, 1);
        else if (splitK <= 4) launch_sgemm_nt_variant<BM, BN, BK, STAGES, THREADS, true, 4>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, actual_beta, C, ldc, strideC, batchCount, 4);
        else if (splitK <= 8) launch_sgemm_nt_variant<BM, BN, BK, STAGES, THREADS, true, 8>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, actual_beta, C, ldc, strideC, batchCount, 8);
        else launch_sgemm_nt_variant<BM, BN, BK, STAGES, THREADS, true, 16>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, actual_beta, C, ldc, strideC, batchCount, 16);
    } else {
        if (splitK <= 1) launch_sgemm_nt_variant<BM, BN, BK, STAGES, THREADS, false, 1>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, actual_beta, C, ldc, strideC, batchCount, 1);
        else if (splitK <= 4) launch_sgemm_nt_variant<BM, BN, BK, STAGES, THREADS, false, 4>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, actual_beta, C, ldc, strideC, batchCount, 4);
        else if (splitK <= 8) launch_sgemm_nt_variant<BM, BN, BK, STAGES, THREADS, false, 8>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, actual_beta, C, ldc, strideC, batchCount, 8);
        else launch_sgemm_nt_variant<BM, BN, BK, STAGES, THREADS, false, 16>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, actual_beta, C, ldc, strideC, batchCount, 16);
    }
}

// 1. cutlass_80_tensorop_s1688gemm_256x128_16x3
extern "C" void mycublasSgemmStridedBatched_nt_256x128_16x3(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK) {
    dispatch_sgemm_nt<256, 128, 16, 3, 256>(handle ? handle->stream : 0, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
}

// 2. cutlass_80_tensorop_s1688gemm_128x256_16x3
extern "C" void mycublasSgemmStridedBatched_nt_128x256_16x3(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK) {
    dispatch_sgemm_nt<128, 256, 16, 3, 256>(handle ? handle->stream : 0, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
}

// 3. cutlass_80_tensorop_s1688gemm_128x128_16x5
extern "C" void mycublasSgemmStridedBatched_nt_128x128_16x5(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK) {
    dispatch_sgemm_nt<128, 128, 16, 5, 128>(handle ? handle->stream : 0, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
}

// 4. cutlass_80_tensorop_s1688gemm_128x128_32x3
extern "C" void mycublasSgemmStridedBatched_nt_128x128_32x3(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK) {
    dispatch_sgemm_nt<128, 128, 32, 3, 128>(handle ? handle->stream : 0, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
}

// 5. cutlass_80_tensorop_s1688gemm_64x64_16x6
extern "C" void mycublasSgemmStridedBatched_nt_64x64_16x6(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK) {
    dispatch_sgemm_nt<64, 64, 16, 6, 64>(handle ? handle->stream : 0, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
}

// 6. cutlass_80_tensorop_s1688gemm_64x64_32x6
extern "C" void mycublasSgemmStridedBatched_nt_64x64_32x6(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK) {
    dispatch_sgemm_nt<64, 64, 32, 6, 64>(handle ? handle->stream : 0, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
}

// 7. cutlass_80_tensorop_s1688gemm_64x128_32x3
extern "C" void mycublasSgemmStridedBatched_nt_64x128_32x3(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK) {
    dispatch_sgemm_nt<64, 128, 32, 3, 128>(handle ? handle->stream : 0, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
}

// 8. cutlass_80_tensorop_s1688gemm_256x64_16x4
extern "C" void mycublasSgemmStridedBatched_nt_256x64_16x4(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK) {
    dispatch_sgemm_nt<256, 64, 16, 4, 128>(handle ? handle->stream : 0, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
}
extern "C" void mycublasSgemmStridedBatched_nt_128x128_16x4(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK) {
    dispatch_sgemm_nt<128, 128, 16, 4, 128>(handle ? handle->stream : 0, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
}

extern "C" void mycublasSgemmStridedBatched_nt_128x64_16x6(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK) {
    dispatch_sgemm_nt<128, 64, 16, 6, 128>(handle ? handle->stream : 0, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
}

// 9. 128x128_16x3 - Added for compatibility/baseline
extern "C" void mycublasSgemmStridedBatched_nt_128x128_16x3(mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float beta, float* C, int ldc, long long int strideC, int batchCount, int splitK) {
    dispatch_sgemm_nt<128, 128, 16, 3, 128>(handle ? handle->stream : 0, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
}
