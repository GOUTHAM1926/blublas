#include "mycublas.h"
#include <cuda_runtime.h>

extern "C" void mycublasSgemm(
    mycublasHandle_t handle,
    mycublasOperation_t transa, mycublasOperation_t transb,
    int M, int N, int K, 
    float alpha, 
    const float *d_A, int lda, 
    const float *d_B, int ldb, 
    float beta, 
    float *d_C, int ldc
)
{
    // Redirect to the strided batched dispatcher with batchCount = 1 and stride = 0
    mycublasSgemmStridedBatched(handle, transa, transb, M, N, K, alpha, d_A, lda, 0, d_B, ldb, 0, beta, d_C, ldc, 0, 1);
}
