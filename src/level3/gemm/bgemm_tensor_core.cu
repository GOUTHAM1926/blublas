#include "mycublas.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

// ============================================================================
// BF16 WMMA KERNEL (Identical structure to FP16)
// BM=128, BN=128, BK=32, Threads=512
// ============================================================================

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int WM = 32;
constexpr int WN = 32;
constexpr int PAD = 8;

__global__ void bgemm_optimized_kernel(
    int M, int N, int K,
    __nv_bfloat16 alpha,
    const __nv_bfloat16* __restrict__ A, int lda,
    const __nv_bfloat16* __restrict__ B, int ldb,
    __nv_bfloat16 beta,
    __nv_bfloat16* __restrict__ C, int ldc)
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int warp_row = warp_id / 4; 
    const int warp_col = warp_id % 4; 
    
    const __nv_bfloat16 *Ap = A; 
    const __nv_bfloat16 *Bp = B; 
    __nv_bfloat16 *Cp = C;       

    __shared__ __nv_bfloat16 As[2][BM][BK + PAD];
    __shared__ __nv_bfloat16 Bs[2][BK][BN + PAD];
   
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af[2]; 
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2]; // Accumulate in FLOAT for BF16

    #pragma unroll
    for(int i=0; i<2; i++) 
        for(int j=0; j<2; j++) 
            wmma::fill_fragment(acc[i][j], 0.0f);

    auto load_tiles = [&](int ko, int idx) {
        for (int i = tid; i < BM * BK; i += 512) {
            int r = i / BK;
            int c = i % BK;
            int global_r = by * BM + r;
            int global_c = ko + c;
            
            __nv_bfloat16 val = __float2bfloat16(0.0f);
            if (global_r < M && global_c < K) {
                val = Ap[global_r * lda + global_c];
            }
            As[idx][r][c] = val;
        }

        for (int i = tid; i < BK * BN; i += 512) {
            int r = i / BN;
            int c = i % BN;
            int global_r = ko + r;
            int global_c = bx * BN + c;
            
            __nv_bfloat16 val = __float2bfloat16(0.0f);
            if (global_r < K && global_c < N) {
                val = Bp[global_r * ldb + global_c];
            }
            Bs[idx][r][c] = val;
        }
    };

    int write_idx = 0;
    load_tiles(0, write_idx);
    __syncthreads();

    // Main Loop
    for (int k = 0; k < K; k += BK) {
        int read_idx = write_idx;
        write_idx = 1 - write_idx;
        
        if (k + BK < K) {
            load_tiles(k + BK, write_idx);
        }

        #pragma unroll
        for (int ks = 0; ks < BK; ks += 16) {
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                wmma::load_matrix_sync(af[i], &As[read_idx][warp_row * WM + i * 16][ks], BK + PAD);
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                     // Note: loading correct slab of B
                     wmma::load_matrix_sync(bf[j], &Bs[read_idx][ks][warp_col * WN + j * 16], BN + PAD);
                     wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

    // Reuse Shared Memory for Output
    // We accumulate in float, so we need a float buffer for staging results?
    // "float* sm = reinterpret_cast<float*>(As);"
    // As in BF16 is 2 bytes. Float is 4 bytes.
    // Capacity check:
    // As size in bytes: 2 * 128 * 40 * 2 = 20480 bytes.
    // We need 16 warps * 256 floats * 4 bytes = 16 * 1024 * 4 = 16384 bytes.
    // It Fits!
    
    float* sm = reinterpret_cast<float*>(As);
    float* warp_sm = sm + warp_id * 256; 

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            wmma::store_matrix_sync(warp_sm, acc[i][j], 16, wmma::mem_row_major);
            
            int global_tile_r = by * BM + warp_row * WM + i * 16;
            int global_tile_c = bx * BN + warp_col * WN + j * 16;
            
            for (int t = 0; t < 8; t++) {
                int tid_in_warp = tid % 32;
                int ele_idx = tid_in_warp + t * 32;
                
                int r = ele_idx / 16;
                int c = ele_idx % 16;
                
                if (global_tile_r + r < M && global_tile_c + c < N) {
                   float res_f = warp_sm[ele_idx];
                   __nv_bfloat16 val = __float2bfloat16(res_f);

                   int addr = (global_tile_r + r) * ldc + (global_tile_c + c);
                   
                   if (beta != __float2bfloat16(0.0f)) {
                       // Convert C to float to accumulate or use bfloat?
                       // Standard GEMM is C = alpha*AB + beta*C.
                       // Usually accumulation C might be float, but input/output C is BF16.
                       // We can do mix:
                       // C_new = alpha * result + beta * C_old.
                       // Using float math for precision? Or bfloat?
                       // Kernel arguments are __nv_bfloat16.
                       // Let's do BF16 FMA.
                       // Actually, generic way:
                       // float c_old = __bfloat162float(Cp[addr]);
                       // float c_new = __bfloat162float(alpha) * res_f + __bfloat162float(beta) * c_old;
                       // Cp[addr] = __float2bfloat16(c_new);
                       // This is more precise.
                       
                       float alpha_f = __bfloat162float(alpha);
                       float beta_f = __bfloat162float(beta);
                       float c_old = __bfloat162float(Cp[addr]);
                       float final_val = alpha_f * res_f + beta_f * c_old;
                       Cp[addr] = __float2bfloat16(final_val);
                   } else {
                       float alpha_f = __bfloat162float(alpha);
                       Cp[addr] = __float2bfloat16(alpha_f * res_f);
                   }
                }
            }
        }
    }
}

extern "C" void mycublasBgemmStridedBatched_dispatcher(mycublasHandle_t handle, mycublasOperation_t transa, mycublasOperation_t transb, int M, int N, int K, const __nv_bfloat16 alpha, const __nv_bfloat16* A, int lda, long long sA, const __nv_bfloat16* B, int ldb, long long sB, const __nv_bfloat16 beta, __nv_bfloat16* C, int ldc, long long sC, int batchCount);

extern "C" void mycublasBgemm(
    mycublasHandle_t handle,
    int M, int N, int K,
    const __nv_bfloat16 alpha,
    const __nv_bfloat16 *A, int lda,
    const __nv_bfloat16 *B, int ldb,
    const __nv_bfloat16 beta,
    __nv_bfloat16 *C, int ldc)
{
    // Preferred path: Ada/Ampere optimized dispatcher
    if (handle && (handle->is_sm89 || handle->sm_count > 0)) {
        mycublasBgemmStridedBatched_dispatcher(handle, MYCUBLAS_OP_N, MYCUBLAS_OP_N, M, N, K, alpha, A, lda, 0, B, ldb, 0, beta, C, ldc, 0, 1);
        return;
    }

    // Fallback: Original WMMA kernel
    cudaStream_t stream = (handle != nullptr) ? handle->stream : 0;
    dim3 block(512);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    bgemm_optimized_kernel<<<grid, block, 0, stream>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
