#include "mycublas.h"
#include <cuda_runtime.h>
#include <unordered_map>
#include "Bgemm_core_template.cuh"

// Helper to launch
template <int BM, int BN, int BK, int STAGES, int THREADS, bool IsAligned, int SplitK>
void launch_bgemm_tn_variant(
    cudaStream_t stream, int M, int N, int K, const __nv_bfloat16 alpha,
    const __nv_bfloat16* A, int lda, long long int strideA,
    const __nv_bfloat16* B, int ldb, long long int strideB,
    const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, 
    int batchCount, int splitK_actual)
{
    using Config = BgemmTileConfig<BM, BN, BK, STAGES, THREADS>;
    static const size_t smem_bytes = STAGES * Config::STAGE_SIZE * sizeof(__nv_bfloat16);
    
    auto set_attr = [](const void* f, size_t b) {
        static std::unordered_map<const void*, bool> done;
        if (!done[f]) { 
            cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)b); 
            done[f] = true; 
        }
    };
    const void* kernel_func = (const void*)bgemm_backward_template_kernel<Config, IsAligned, SplitK, BgemmLayout::TN>;
    set_attr(kernel_func, smem_bytes);

    const int gx = (N + BN - 1) / BN, gy = (M + BM - 1) / BM;
    dim3 grid(gx, gy, batchCount * splitK_actual);
    
    bgemm_backward_template_kernel<Config, IsAligned, SplitK, BgemmLayout::TN>
        <<<grid, THREADS, smem_bytes, stream>>>(
            M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount
        );
}

template <int BM, int BN, int BK, int STAGES, int THREADS>
void dispatch_bgemm_tn(
    cudaStream_t stream, int M, int N, int K, const __nv_bfloat16 alpha,
    const __nv_bfloat16* A, int lda, long long int strideA,
    const __nv_bfloat16* B, int ldb, long long int strideB,
    const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, 
    int batchCount, int splitK)
{
    const bool aligned = (((size_t)A & 15) == 0) && ((lda & 7) == 0) && 
                         (((size_t)B & 15) == 0) && ((ldb & 7) == 0);
                         
    if (splitK > 1) {
        dim3 s_grid((N+31)/32, (M+31)/32, batchCount);
        bgemm_scale_template_kernel<BgemmTileConfig<BM,BN,BK,STAGES,THREADS>>
            <<<s_grid, dim3(32,32), 0, stream>>>(C, beta, M, N, ldc, strideC, batchCount);
    }
    __nv_bfloat16 actual_beta = (splitK > 1) ? __float2bfloat16(0.0f) : beta;

    if (aligned) {
        if (splitK <= 1) launch_bgemm_tn_variant<BM, BN, BK, STAGES, THREADS, true, 1>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, actual_beta, C, ldc, strideC, batchCount, 1);
        else if (splitK <= 4) launch_bgemm_tn_variant<BM, BN, BK, STAGES, THREADS, true, 4>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, actual_beta, C, ldc, strideC, batchCount, 4);
        else launch_bgemm_tn_variant<BM, BN, BK, STAGES, THREADS, true, 16>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, actual_beta, C, ldc, strideC, batchCount, 16);
    } else {
        if (splitK <= 1) launch_bgemm_tn_variant<BM, BN, BK, STAGES, THREADS, false, 1>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, actual_beta, C, ldc, strideC, batchCount, 1);
        else if (splitK <= 4) launch_bgemm_tn_variant<BM, BN, BK, STAGES, THREADS, false, 4>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, actual_beta, C, ldc, strideC, batchCount, 4);
        else launch_bgemm_tn_variant<BM, BN, BK, STAGES, THREADS, false, 16>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, actual_beta, C, ldc, strideC, batchCount, 16);
    }
}

extern "C" void mycublasBgemmStridedBatched_tn_256x128_32x3(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK) {
    dispatch_bgemm_tn<256, 128, 32, 4, 256>(handle ? handle->stream : 0, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
}

extern "C" void mycublasBgemmStridedBatched_tn_128x256_32x3(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK) {
    dispatch_bgemm_tn<128, 256, 32, 4, 256>(handle ? handle->stream : 0, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
}

extern "C" void mycublasBgemmStridedBatched_tn_128x128_32x5(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK) {
    dispatch_bgemm_tn<128, 128, 32, 5, 128>(handle ? handle->stream : 0, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
}

extern "C" void mycublasBgemmStridedBatched_tn_128x128_64x3(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK) {
    dispatch_bgemm_tn<128, 128, 64, 3, 128>(handle ? handle->stream : 0, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
}

extern "C" void mycublasBgemmStridedBatched_tn_64x64_32x6(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK) {
    dispatch_bgemm_tn<64, 64, 32, 6, 64>(handle ? handle->stream : 0, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
}

extern "C" void mycublasBgemmStridedBatched_tn_64x64_64x6(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK) {
    dispatch_bgemm_tn<64, 64, 64, 6, 64>(handle ? handle->stream : 0, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
}

extern "C" void mycublasBgemmStridedBatched_tn_64x128_64x3(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK) {
    dispatch_bgemm_tn<64, 128, 64, 3, 128>(handle ? handle->stream : 0, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
}

extern "C" void mycublasBgemmStridedBatched_tn_256x64_32x4(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK) {
    dispatch_bgemm_tn<256, 64, 32, 4, 128>(handle ? handle->stream : 0, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
}

extern "C" void mycublasBgemmStridedBatched_tn_128x128_32x4(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK) {
    dispatch_bgemm_tn<128, 128, 32, 4, 128>(handle ? handle->stream : 0, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
}

extern "C" void mycublasBgemmStridedBatched_tn_128x64_32x6(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK) {
    dispatch_bgemm_tn<128, 64, 32, 6, 128>(handle ? handle->stream : 0, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
}

extern "C" void mycublasBgemmStridedBatched_tn_128x128_32x3(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK) {
    dispatch_bgemm_tn<128, 128, 32, 5, 128>(handle ? handle->stream : 0, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK);
}
