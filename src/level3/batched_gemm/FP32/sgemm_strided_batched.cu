#include "mycublas.h"
#include <cuda_runtime.h>

extern "C" void mycublasSgemmStridedBatched(
    mycublasHandle_t handle,
    mycublasOperation_t transA, mycublasOperation_t transB,
    int M, int N, int K,
    const float alpha,
    const float *A, int lda, long long int strideA,
    const float *B, int ldb, long long int strideB,
    const float beta,
    float *C, int ldc, long long int strideC,
    int batchCount)
{
    if (transA == MYCUBLAS_OP_T && transB == MYCUBLAS_OP_N) {
        mycublasSgemmStridedBatched_tn_dispatch(handle, M, N, K, alpha, A, (lda == 0 ? M : lda), strideA, B, (ldb == 0 ? N : ldb), strideB, beta, C, (ldc == 0 ? N : ldc), strideC, batchCount);
    } else if (transA == MYCUBLAS_OP_N && transB == MYCUBLAS_OP_T) {
        mycublasSgemmStridedBatched_nt_dispatch(handle, M, N, K, alpha, A, (lda == 0 ? K : lda), strideA, B, (ldb == 0 ? K : ldb), strideB, beta, C, (ldc == 0 ? N : ldc), strideC, batchCount);
    } else {
        mycublasSgemmStridedBatched_nn_dispatch(handle, M, N, K, alpha, A, (lda == 0 ? K : lda), strideA, B, (ldb == 0 ? N : ldb), strideB, beta, C, (ldc == 0 ? N : ldc), strideC, batchCount);
    }
}
