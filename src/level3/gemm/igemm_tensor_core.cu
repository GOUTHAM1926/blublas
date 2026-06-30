#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>
#include "mycublas.h"

using namespace nvcuda;

// WMMA Configuration for Int8
// Ampere supports 16x16x16 for Int8
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void wmma_igemm_kernel(
    int M, int N, int K,
    int32_t alpha,
    const int8_t *__restrict__ A, int lda,
    const int8_t *__restrict__ B, int ldb,
    int32_t beta,
    int32_t *__restrict__ C, int ldc)
{
    // Block: 1 Warp
    int globalWarpM = blockIdx.y;
    int globalWarpN = blockIdx.x;
    
    // Fragments for Int8
    // Layout: A must be Row Major, B must be Col Major for optimal WMMA?
    // Using experimental::precision::s8 for signed int8
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::col_major> b_frag; 
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> c_frag; 
    
    wmma::fill_fragment(c_frag, 0);
    
    for (int k = 0; k < K; k += WMMA_K) {
        const int8_t* a_ptr = A + (globalWarpM * WMMA_M) + (k * lda); 
        const int8_t* b_ptr = B + (k) + (globalWarpN * WMMA_N * ldb);
        
        wmma::load_matrix_sync(a_frag, a_ptr, lda);
        wmma::load_matrix_sync(b_frag, b_ptr, ldb);
        
        // mma_sync for int8 needs specific precision tagging in some versions, 
        // but default template deduction usually works if frag types are correct.
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // Store
    int cRow = globalWarpM * WMMA_M;
    int cCol = globalWarpN * WMMA_N;
    int32_t* c_ptr = C + cRow + cCol * ldc;
    
    wmma::store_matrix_sync(c_ptr, c_frag, ldc, wmma::mem_col_major);
}

extern "C" void mycublasIgemm(
    mycublasHandle_t handle,
    int M, int N, int K,
    const int32_t alpha,
    const int8_t *d_A, int lda,
    const int8_t *d_B, int ldb,
    const int32_t beta,
    int32_t *d_C, int ldc)
{
    cudaStream_t stream = (handle != nullptr) ? handle->stream : 0;
    dim3 block(32, 1);
    dim3 grid((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
    
    wmma_igemm_kernel<<<grid, block, 0, stream>>>(M, N, K, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
}
