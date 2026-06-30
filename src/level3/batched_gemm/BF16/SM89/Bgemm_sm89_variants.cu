#include "Bgemm_core_template.cuh"
#include "mycublas.h"
#include <unordered_map>

// Helper to launch
template <int BM, int BN, int BK, int STAGES, int THREADS, bool IsAligned, int SplitK, BgemmLayout Layout>
void launch_bgemm_variant(
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
    const void* kernel_func = (const void*)bgemm_backward_template_kernel<Config, IsAligned, SplitK, Layout>;
    set_attr(kernel_func, smem_bytes);

    const int gx = (N + BN - 1) / BN, gy = (M + BM - 1) / BM;
    dim3 grid(gx, gy, batchCount * splitK_actual);
    
    bgemm_backward_template_kernel<Config, IsAligned, SplitK, Layout>
        <<<grid, THREADS, smem_bytes, stream>>>(
            M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount
        );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Bgemm kernel launch failed: %s (smem=%zu bytes)\n", cudaGetErrorString(err), smem_bytes);
    }
}

template <int BM, int BN, int BK, int STAGES, int THREADS, BgemmLayout Layout>
void dispatch_bgemm_layout(
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
        if (splitK <= 1) launch_bgemm_variant<BM, BN, BK, STAGES, THREADS, true, 1, Layout>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, actual_beta, C, ldc, strideC, batchCount, 1);
        else if (splitK <= 4) launch_bgemm_variant<BM, BN, BK, STAGES, THREADS, true, 4, Layout>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, actual_beta, C, ldc, strideC, batchCount, 4);
        else launch_bgemm_variant<BM, BN, BK, STAGES, THREADS, true, 16, Layout>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, actual_beta, C, ldc, strideC, batchCount, 16);
    } else {
        if (splitK <= 1) launch_bgemm_variant<BM, BN, BK, STAGES, THREADS, false, 1, Layout>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, actual_beta, C, ldc, strideC, batchCount, 1);
        else if (splitK <= 4) launch_bgemm_variant<BM, BN, BK, STAGES, THREADS, false, 4, Layout>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, actual_beta, C, ldc, strideC, batchCount, 4);
        else launch_bgemm_variant<BM, BN, BK, STAGES, THREADS, false, 16, Layout>(stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, actual_beta, C, ldc, strideC, batchCount, 16);
    }
}

#define BGEMM_STRIDED_BATCHED_INST(LAYOUT, LAYER_NAME, BM, BN, BK, STAGES, THREADS) \
extern "C" void mycublasBgemmStridedBatched_##LAYER_NAME##_##BM##x##BN##_##BK##_##STAGES(mycublasHandle_t handle, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long int strideA, const __nv_bfloat16* B, int ldb, long long int strideB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long int strideC, int batchCount, int splitK) { \
    dispatch_bgemm_layout<BM, BN, BK, STAGES, THREADS, BgemmLayout::LAYOUT>(handle ? handle->stream : 0, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount, splitK); \
}

// NN
BGEMM_STRIDED_BATCHED_INST(NN, nn, 256, 128, 32, 4, 256)
BGEMM_STRIDED_BATCHED_INST(NN, nn, 256, 128, 32, 5, 256)
BGEMM_STRIDED_BATCHED_INST(NN, nn, 128, 256, 32, 4, 256)
BGEMM_STRIDED_BATCHED_INST(NN, nn, 128, 256, 32, 5, 256)
BGEMM_STRIDED_BATCHED_INST(NN, nn, 128, 128, 32, 6, 128)

// NT
BGEMM_STRIDED_BATCHED_INST(NT, nt, 256, 128, 32, 4, 256)
BGEMM_STRIDED_BATCHED_INST(NT, nt, 256, 128, 32, 5, 256)
BGEMM_STRIDED_BATCHED_INST(NT, nt, 128, 256, 32, 4, 256)
BGEMM_STRIDED_BATCHED_INST(NT, nt, 128, 256, 32, 5, 256)
BGEMM_STRIDED_BATCHED_INST(NT, nt, 128, 128, 32, 6, 128)

// TN
BGEMM_STRIDED_BATCHED_INST(TN, tn, 256, 128, 32, 4, 256)
BGEMM_STRIDED_BATCHED_INST(TN, tn, 256, 128, 32, 5, 256)
BGEMM_STRIDED_BATCHED_INST(TN, tn, 128, 256, 32, 4, 256)
BGEMM_STRIDED_BATCHED_INST(TN, tn, 128, 256, 32, 5, 256)
BGEMM_STRIDED_BATCHED_INST(TN, tn, 128, 128, 32, 6, 128)
