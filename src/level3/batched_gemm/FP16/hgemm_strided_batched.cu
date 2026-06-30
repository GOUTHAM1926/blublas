#include "mycublas.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>


using namespace nvcuda;


// FP16 Strided Batched GEMM using WMMA and Async Copy
// Grid-level batching: blockIdx.z represents batch index
// Tiling: BM=128, BN=128, BK=32 (Standard for FP16)
// Threads: 128 (4 warps)
// Pipeline: 3-stage async


#define BM 128
#define BN 128
#define BK 32


#define THREADS_PER_BLOCK 128


__device__ __forceinline__ void cp_async_16(void* smem_ptr, const void* glob_ptr) {
   uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
   asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(glob_ptr));
}


__global__ void hgemm_async_strided_batched_kernel(
   int M, int N, int K,
   __half alpha,
   const __half* __restrict__ A, int lda, long long int strideA,
   const __half* __restrict__ B, int ldb, long long int strideB,
   __half beta,
   __half* __restrict__ C, int ldc, long long int strideC)
{
   // Batch index
   const int batch_idx = blockIdx.z;
   const __half* A_batch = A + batch_idx * strideA;
   const __half* B_batch = B + batch_idx * strideB;
   __half* C_batch = C + batch_idx * strideC;


   // Shared Memory for 3 stages
   // Size: 3 * (TileA + TileB)
   extern __shared__ __half s_mem[];
   __half* As = s_mem;
   __half* Bs = s_mem + 3 * BM * BK;


   // WMMA Accumulators: 4x4 fragments per warp
   wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][4];
  
   #pragma unroll
   for(int i=0; i<4; i++)
       for(int j=0; j<4; j++)
           wmma::fill_fragment(acc[i][j], 0.0f);


   const int bx = blockIdx.x;
   const int by = blockIdx.y;
   const int tid = threadIdx.x;
   const int wid = tid / 32;
   const int warp_y = wid / 2;
   const int warp_x = wid % 2;


   const __half* dA = A_batch + (by * BM) * lda;
   const __half* dB = B_batch + (bx * BN);


   // Lambda to load tiles
   auto load_tile = [&](int k_off, int buf_idx) {
       // Load A: 128x32
       #pragma unroll
       for (int i = 0; i < 4; i++) {
           int vec_idx = tid + i * 128;
           int r = vec_idx >> 2;
           int c = (vec_idx & 3) << 3;


           __half* sp = &As[buf_idx * BM * BK + r * BK + c];
          
           if (by*BM + r < M && k_off + c < K) {
               const __half* src = &dA[r * lda + k_off + c];
               if (((reinterpret_cast<uintptr_t>(src) & 15) == 0) && (k_off + c + 8 <= K)) {
                   cp_async_16(sp, src);
               } else {
                   #pragma unroll
                   for(int e=0; e<8; e++) {
                       sp[e] = (k_off + c + e < K) ? src[e] : __float2half(0.0f);
                   }
               }
           } else {
               *(int4*)sp = {0,0,0,0}; // Zero out 16 bytes
           }
       }


       // Load B: 32x128
       #pragma unroll
       for (int i = 0; i < 4; i++) {
           int vec_idx = tid + i * 128;
           int r = vec_idx >> 4;
           int c = (vec_idx & 15) << 3;


            __half* sp = &Bs[buf_idx * BK * BN + r * BN + c];
           
            if (k_off + r < K && bx*BN + c < N) {
                const __half* src = &dB[(k_off + r) * ldb + c];
                if (((reinterpret_cast<uintptr_t>(src) & 15) == 0) && (bx*BN + c + 8 <= N)) {
                    cp_async_16(sp, src);
                } else {
                    #pragma unroll
                    for(int e=0; e<8; e++) {
                        sp[e] = (bx*BN + c + e < N) ? src[e] : __float2half(0.0f);
                    }
                }
            } else {
                *(int4*)sp = {0,0,0,0};
            }
       }
       asm volatile("cp.async.commit_group;\n");
   };


   // Prologue
   int write_idx = 0;
   if (0 < K) load_tile(0, 0);
   write_idx = (write_idx + 1) % 3;
   if (BK < K) load_tile(BK, 1);
   write_idx = (write_idx + 1) % 3;


   // Main Loop
   int read_idx = 0;
   for (int k = 0; k < K; k += BK) {
       asm volatile("cp.async.wait_group 1;\n");
       __syncthreads();


       int cur_read = read_idx;
       read_idx = (read_idx + 1) % 3;
      
       int prefetch_k = k + 2 * BK;
       if (prefetch_k < K) {
           load_tile(prefetch_k, write_idx);
           write_idx = (write_idx + 1) % 3;
       }


       // Compute
       #pragma unroll
       for (int ks = 0; ks < BK; ks += 16) {
           wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[4];
           wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb[4];


           #pragma unroll
           for (int i = 0; i < 4; i++) {
               int row = warp_y * 64 + i * 16;
               wmma::load_matrix_sync(fa[i], &As[cur_read*BM*BK + row*BK + ks], BK);
           }


           #pragma unroll
           for (int j = 0; j < 4; j++) {
               int col = warp_x * 64 + j * 16;
               wmma::load_matrix_sync(fb[j], &Bs[cur_read*BK*BN + ks*BN + col], BN);
           }


           #pragma unroll
           for (int i = 0; i < 4; i++) {
               #pragma unroll
               for (int j = 0; j < 4; j++) {
                   wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
               }
           }
       }
   }
  
   // Store Results
   asm volatile("cp.async.wait_group 0;\n");
   __syncthreads();


   float* stage_f = reinterpret_cast<float*>(As);
   float f_alpha = __half2float(alpha);
   float f_beta = __half2float(beta);


   #pragma unroll
   for (int i = 0; i < 4; i++) {
       for (int j = 0; j < 4; j++) {
           int global_r = by * BM + warp_y * 64 + i * 16;
           int global_c = bx * BN + warp_x * 64 + j * 16;
          
           float* ws = stage_f + wid * 256;
           wmma::store_matrix_sync(ws, acc[i][j], 16, wmma::mem_row_major);
           __syncwarp();
          
           for (int e = 0; e < 8; e++) {
               int tid_idx = tid % 32;
               int linear_idx = tid_idx * 8 + e;
               int r = linear_idx / 16;
               int c = linear_idx % 16;
              
               if (global_r + r < M && global_c + c < N) {
                   float val = ws[linear_idx];
                   int addr = (global_r + r) * ldc + (global_c + c);
                  
                   __half res;
                   if (beta == __float2half(0.0f)) {
                        res = __float2half(f_alpha * val);
                   } else {
                        res = __float2half(f_alpha * val + f_beta * __half2float(C_batch[addr]));
                   }
                   C_batch[addr] = res;
               }
           }
           __syncwarp();
       }
   }
}


extern "C" void mycublasHgemmStridedBatched_dispatcher(
    mycublasHandle_t handle,
    mycublasOperation_t transA, mycublasOperation_t transB,
    int M, int N, int K,
    const __half alpha,
    const __half *A, int lda, long long int strideA,
    const __half *B, int ldb, long long int strideB,
    const __half beta,
    __half *C, int ldc, long long int strideC,
    int batchCount);

extern "C" void mycublasHgemmStridedBatched(
    mycublasHandle_t handle,
    mycublasOperation_t transA, mycublasOperation_t transB,
    int M, int N, int K,
    const __half alpha,
    const __half *A, int lda, long long int strideA,
    const __half *B, int ldb, long long int strideB,
    const __half beta,
    __half *C, int ldc, long long int strideC,
    int batchCount)
{
    mycublasHgemmStridedBatched_dispatcher(handle, transA, transB, M, N, K, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}
