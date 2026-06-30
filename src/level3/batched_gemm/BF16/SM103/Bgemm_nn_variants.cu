// Bgemm_forward_nn_cutlass_variants.cu
// Blackwell Ultra (SM103a / B300) BF16 Batched GEMM — NN layout, 2-CTA cluster variants
//
// Implements the cluster-variant entry points using the unified template.

#include "mycublas.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "Bgemm_sm103_cluster_template.cuh"

extern "C" {

void mycublasBgemmSM103_bf16_nn_cluster_128x128x64_template(
    mycublasHandle_t handle,
    int M, int N, int K,
    float alpha,
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    float beta,
          __nv_bfloat16* C,
    int batchCount)
{
    cudaStream_t stream = handle ? handle->stream : 0;
    launch_bgemm_sm103_cluster<Sm103Config_128x128x64, Sm103Layout::NN>(
        A, B, C, M, N, K, batchCount, alpha, beta, stream);
}

void mycublasBgemmSM103_bf16_nn_cluster_256x256x64_template(
    mycublasHandle_t handle,
    int M, int N, int K,
    float alpha,
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    float beta,
          __nv_bfloat16* C,
    int batchCount)
{
    cudaStream_t stream = handle ? handle->stream : 0;
    launch_bgemm_sm103_cluster<Sm103Config_256x256x64, Sm103Layout::NN>(
        A, B, C, M, N, K, batchCount, alpha, beta, stream);
}

} // extern "C"
