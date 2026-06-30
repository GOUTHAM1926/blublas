#include "mycublas.h"
#include <mma.h>
#include <cuda_runtime.h>
#include <stdio.h>

using namespace nvcuda;

// V43 (Revert): 128x128x16 with 3-stage WMMA pipeline
#define BM 128
#define BN 128
#define BK 16

#define THREADS_PER_BLOCK 128
#define WARPS_PER_BLOCK 4

__device__ __forceinline__ void cp_async_16(void* smem_ptr, const void* glob_ptr) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(glob_ptr));
}

__global__ void wmma_tf32_gemm(
    const float* __restrict__ A, int lda,
    const float* __restrict__ B, int ldb,
    float* __restrict__ C, int ldc,
    int M, int N, int K,
    float alpha, float beta
) {
    extern __shared__ float s_mem[];
    float* As = s_mem;
    float* Bs = s_mem + 3 * BM * BK;

    wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) wmma::fill_fragment(acc[i][j], 0.0f);

    const int bx = blockIdx.x, by = blockIdx.y;
    const int tid = threadIdx.x, wid = tid / 32;
    const int warp_y = wid / 2, warp_x = wid % 2;
    
    const float* dA = A + (by * BM) * lda;
    const float* dB = B + (bx * BN);

    auto load_tile = [&](int k_off, int buf_idx) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = tid * 4 + i * 512;
            int r = idx / BK, c = idx % BK;
            void* sp = &As[buf_idx * BM * BK + r * BK + c];
            if (by*BM+r < M && k_off+c < K) {
                const float* src = &dA[r*lda + k_off + c];
                if (((reinterpret_cast<uintptr_t>(src) & 15) == 0) && (k_off + c + 4 <= K)) {
                    cp_async_16(sp, src);
                } else {
                    float* sp_f = (float*)sp;
                    #pragma unroll
                    for(int e=0; e<4; e++) {
                        sp_f[e] = (k_off+c+e < K) ? src[e] : 0.0f;
                    }
                }
            } else *(float4*)sp = {0,0,0,0};
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = tid * 4 + i * 512;
            int r = idx / BN, bc = idx % BN;
            void* sp = &Bs[buf_idx * BK * BN + r * BN + bc];
            if (k_off+r < K && bx*BN+bc < N) {
                const float* src = &dB[(k_off + r)*ldb + bc];
                if (((reinterpret_cast<uintptr_t>(src) & 15) == 0) && (bx * BN + bc + 4 <= N)) {
                    cp_async_16(sp, src);
                } else {
                    float* sp_f = (float*)sp;
                    #pragma unroll
                    for(int e=0; e<4; e++) {
                        sp_f[e] = (bx*BN+bc+e < N) ? src[e] : 0.0f;
                    }
                }
            } else *(float4*)sp = {0,0,0,0};
        }
        asm volatile("cp.async.commit_group;\n");
    };

    int write_idx = 0;
    if (0 < K) load_tile(0, 0);
    write_idx = (write_idx + 1) % 3;
    if (BK < K) load_tile(BK, 1);
    write_idx = (write_idx + 1) % 3;

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

        #pragma unroll
        for (int ks = 0; ks < 16; ks += 8) {
            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> fa[4];
            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> fb[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) 
                wmma::load_matrix_sync(fa[i], &As[cur_read*BM*BK + (warp_y*64+i*16)*BK + ks], BK);
            #pragma unroll
            for (int j = 0; j < 4; j++) 
                wmma::load_matrix_sync(fb[j], &Bs[cur_read*BK*BN + ks*BN + (warp_x*64+j*16)], BN);
            #pragma unroll
            for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) 
                wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
        }
    }

    __syncthreads();
    float* stage = As; 
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int row = by * BM + warp_y * 64 + i * 16, col = bx * BN + warp_x * 64 + j * 16;
            if (row < M && col < N) {
                if (alpha == 1.0f && beta == 0.0f) wmma::store_matrix_sync(C + row * ldc + col, acc[i][j], ldc, wmma::mem_row_major);
                else {
                    float* ws = stage + wid * 256;
                    wmma::store_matrix_sync(ws, acc[i][j], 16, wmma::mem_row_major);
                    __syncwarp();
                    for(int e=0; e<8; e++) {
                        int idx = (tid%32)*8 + e, r=idx/16, c=idx%16;
                        if(row+r<M && col+c<N) C[(row+r)*ldc+col+c] = alpha*ws[idx] + beta*C[(row+r)*ldc+col+c];
                    }
                    __syncwarp();
                }
            }
        }
    }
}


extern "C" void mycublasSgemm_TensorCore_Explicit(
    mycublasHandle_t handle, 
    mycublasOperation_t transa, 
    mycublasOperation_t transb, 
    int M, int N, int K, 
    float alpha, const float *A, int lda, 
    const float *B, int ldb, 
    float beta, float *C, int ldc, 
    bool use_tf32) 
{
    if (use_tf32) {
        // 1. Get the stream from your handle
        cudaStream_t stream = (handle != nullptr) ? handle->stream : 0;

        // 2. Calculate shared memory size (3 stages of As and Bs)
        size_t sz = (3 * BM * BK + 3 * BK * BN) * sizeof(float);

        // 3. Set dynamic shared memory limit once
        static bool init = false;
        if (!init) {
            cudaFuncSetAttribute(wmma_tf32_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sz);
            init = true;
        }

        // 4. Define Grid and Block
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 block(THREADS_PER_BLOCK);

        // 5. LAUNCH the kernel with <<< >>>
        wmma_tf32_gemm<<<grid, block, sz, stream>>>(
            A, lda, B, ldb, C, ldc, M, N, K, alpha, beta
        );
    } else {
        // Standard cuBLAS fallback
        mycublasSgemm(handle, transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

