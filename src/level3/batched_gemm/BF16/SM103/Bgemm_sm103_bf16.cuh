#pragma once
// Blackwell Ultra (SM103a / B300) BF16 Batched GEMM declarations.
// Include this header in the SM103 dispatcher or any translation unit that
// needs to call these kernels directly.

#include "mycublas.h"

#ifdef __cplusplus
extern "C" {
#endif

// NT layout: C[b,M,N] = alpha * A[b,M,K] × B[b,N,K]^T + beta * C[b,M,N]
//
// B must be provided in [batchCount, N, K] row-major order.
// Tile 128×128×64.  M % 128 == 0, N % 128 == 0, K % 64 == 0.
// Requires SM103a (B300). Uses tcgen05.mma.cta_group::1.kind::f16.
void mycublasBgemmSM103_bf16_nt_128x128x64(
    mycublasHandle_t handle,
    int M, int N, int K,
    float alpha,
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    float beta,
          __nv_bfloat16* C,
    int batchCount);

// Same kernel, accepts explicit batch strides (must be packed row-major).
void mycublasBgemmSM103_bf16_nt_strided(
    mycublasHandle_t handle,
    int M, int N, int K,
    float alpha,
    const __nv_bfloat16* A, long long strideA,
    const __nv_bfloat16* B, long long strideB,
    float beta,
          __nv_bfloat16* C, long long strideC,
    int batchCount);

// NN layout: C[b,M,N] = alpha * A[b,M,K] × B[b,K,N] + beta * C[b,M,N]
//
// B must be provided in [batchCount, K, N] row-major order.
void mycublasBgemmSM103_bf16_nn_128x128x64(
    mycublasHandle_t handle,
    int M, int N, int K,
    float alpha,
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    float beta,
          __nv_bfloat16* C,
    int batchCount);

void mycublasBgemmSM103_bf16_nn_strided(
    mycublasHandle_t handle,
    int M, int N, int K,
    float alpha,
    const __nv_bfloat16* A, long long strideA,
    const __nv_bfloat16* B, long long strideB,
    float beta,
          __nv_bfloat16* C, long long strideC,
    int batchCount);

// Cluster kernels (cta_group::2):
void mycublasBgemmSM103_bf16_nn_cluster_128x128x64(
    mycublasHandle_t handle,
    int M, int N, int K,
    float alpha,
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    float beta,
          __nv_bfloat16* C,
    int batchCount);

void mycublasBgemmSM103_bf16_nn_cluster_256x256x64(
    mycublasHandle_t handle,
    int M, int N, int K,
    float alpha,
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    float beta,
          __nv_bfloat16* C,
    int batchCount);

// NT layout cluster kernels (cta_group::2):
void mycublasBgemmSM103_bf16_nt_cluster_128x128x64(
    mycublasHandle_t handle,
    int M, int N, int K,
    float alpha,
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    float beta,
          __nv_bfloat16* C,
    int batchCount);

void mycublasBgemmSM103_bf16_nt_cluster_256x256x64(
    mycublasHandle_t handle,
    int M, int N, int K,
    float alpha,
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    float beta,
          __nv_bfloat16* C,
    int batchCount);

// TN layout cluster kernels (cta_group::2):
void mycublasBgemmSM103_bf16_tn_cluster_128x128x64(
    mycublasHandle_t handle,
    int M, int N, int K,
    float alpha,
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    float beta,
          __nv_bfloat16* C,
    int batchCount);

void mycublasBgemmSM103_bf16_tn_cluster_256x256x64(
    mycublasHandle_t handle,
    int M, int N, int K,
    float alpha,
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    float beta,
          __nv_bfloat16* C,
    int batchCount);

#ifdef __cplusplus
}
#endif
