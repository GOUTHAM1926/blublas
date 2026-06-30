// Bgemm_sm103_dispatcher.cu
// Routes BF16 batched GEMM requests to the cta_group::1 tcgen05 kernel.
//
// Target: sm_100a (B100/B200) natively; sm_103a (B300) via JIT from sm_100a PTX.
//
// Per PTX ISA 9.3 p.757, tcgen05.mma.kind::f16 is valid for:
//   sm_100a, sm_101a/sm_110a, sm_100f/sm_110f family (which covers B300 via JIT).
//   It is NOT valid as a sm_103a NATIVE binary target — compile with
//   code=compute_100a (PTX) so B300 JIT-compiles through the family path.
//
// Alignment requirements:
//   M % 128, N % 128, K % 64 — must hold before dispatch.
//   B argument must be in [batch, N, K] layout (transposed) for the NT kernel.

#include "Bgemm_sm103_bf16.cuh"
#include "mycublas.h"
#include <cuda_bf16.h>

static inline bool sm103_tile_aligned(int M, int N, int K)
{
    // Require M divisible by 2*128=256 — the cluster covers 2 BM-row blocks.
    // Sizes where M%256!=0 fall through to SM89 kernels.
    return (M % 256 == 0) && (N % 128 == 0) && (K % 64 == 0);
}

// Returns true if the kernel was dispatched, false if it fell through.
extern "C" bool mycublasBgemmSM103_dispatch_nt(
    mycublasHandle_t handle,
    int M, int N, int K,
    float alpha,
    const __nv_bfloat16* A, long long strideA,
    const __nv_bfloat16* B, long long strideB,
    float beta,
          __nv_bfloat16* C, long long strideC,
    int batchCount)
{
    if (!sm103_tile_aligned(M, N, K))
        return false;
    if (strideA != (long long)M * K ||
        strideB != (long long)N * K ||
        strideC != (long long)M * N)
        return false;

    mycublasBgemmSM103_bf16_nt_cluster_128x128x64(
        handle, M, N, K, alpha, A, B, beta, C, batchCount);
    return true;
}

extern "C" bool mycublasBgemmSM103_dispatch_nn(
    mycublasHandle_t handle,
    int M, int N, int K,
    float alpha,
    const __nv_bfloat16* A, long long strideA,
    const __nv_bfloat16* B, long long strideB,
    float beta,
          __nv_bfloat16* C, long long strideC,
    int batchCount)
{
    if (!sm103_tile_aligned(M, N, K))
        return false;
    if (strideA != (long long)M * K ||
        strideB != (long long)K * N ||
        strideC != (long long)M * N)
        return false;

    mycublasBgemmSM103_bf16_nn_cluster_128x128x64(
        handle, M, N, K, alpha, A, B, beta, C, batchCount);
    return true;
}

extern "C" bool mycublasBgemmSM103_dispatch_tn(
    mycublasHandle_t handle,
    int M, int N, int K,
    float alpha,
    const __nv_bfloat16* A, long long strideA,
    const __nv_bfloat16* B, long long strideB,
    float beta,
          __nv_bfloat16* C, long long strideC,
    int batchCount)
{
    if (!sm103_tile_aligned(M, N, K))
        return false;
    if (strideA != (long long)K * M ||
        strideB != (long long)K * N ||
        strideC != (long long)M * N)
        return false;

    mycublasBgemmSM103_bf16_tn_cluster_128x128x64(
        handle, M, N, K, alpha, A, B, beta, C, batchCount);
    return true;
}
