#include "mycublas.h"
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>

using namespace nvcuda;

// FP64 Strided Batched GEMM using DMMA
// Grid-level batching: blockIdx.z represents batch index

// DMMA Config
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;   // Larger K for better memory reuse
constexpr int WM = 16;   // Warp M
constexpr int WN = 32;   // Warp N
constexpr int PAD = 4;   // Padding for doubles (8 bytes), alignment check
constexpr int DMMA_M = 8;
constexpr int DMMA_N = 8;
constexpr int DMMA_K = 4;

__global__ void dgemm_strided_batched_kernel(
    int M, int N, int K,
    double alpha,
    const double* __restrict__ A, int lda, long long int strideA,
    const double* __restrict__ B, int ldb, long long int strideB,
    double beta,
    double* __restrict__ C, int ldc, long long int strideC)
{
    // Batch index from grid Z-dimension
    const int batch_idx = blockIdx.z;
    
    // Compute batch-specific pointers
    const double* A_batch = A + batch_idx * strideA;
    const double* B_batch = B + batch_idx * strideB;
    double* C_batch = C + batch_idx * strideC;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int warp_row = warp_id / 2; // 0..3 (4 rows of warps? No, 256 threads = 8 warps)
                                      // 8 Warps for 64x64 output?
                                      // 64x64 = 4096 elements.
                                      // WarpTile 16x32 = 512 elements.
                                      // 4096 / 512 = 8 Warps.
                                      // Grid: 4x2 warps.
                                      // warp_row = warp_id / 2 (0..3)
                                      // warp_col = warp_id % 2 (0..1)
    const int warp_col = warp_id % 2; 

    // Shared Memory Double Buffering
    // As[2][BM][BK+PAD]
    // 2 * 64 * 20 * 8 bytes = 20KB. OK.
    __shared__ double As[2][BM][BK + PAD];
    __shared__ double Bs[2][BK][BN + PAD];
    
    // Fragments
    // WarpTile 16x32.
    // DMMA 8x8x4.
    // M dim: 16 / 8 = 2.
    // N dim: 32 / 8 = 4.
    // Total 2x4 = 8 mma ops.
    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> a_frags[2];
    wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> b_frags[4];
    wmma::fragment<wmma::accumulator, 8, 8, 4, double> acc[2][4];

    // Init Acc
    #pragma unroll
    for(int i=0; i<2; i++) 
        for(int j=0; j<4; j++) 
            wmma::fill_fragment(acc[i][j], 0.0);

    // Load Pipeline
    auto load_tiles = [&](int ko, int idx) {
        // Load A: BM x BK (64 x 16). 1024 elements.
        // Threads 256. 4 elems/thread.
        for (int i = tid; i < BM * BK; i += 256) {
            int r = i / BK;
            int c = i % BK;
            int gr = by * BM + r;
            int gc = ko + c;
            As[idx][r][c] = (gr < M && gc < K) ? A_batch[gr*lda + gc] : 0.0;
        }
        // Load B: BK x BN (16 x 64). 1024 elements.
        for (int i = tid; i < BK * BN; i += 256) {
            int r = i / BN;
            int c = i % BN;
            int gr = ko + r;
            int gc = bx * BN + c;
            Bs[idx][r][c] = (gr < K && gc < N) ? B_batch[gr*ldb + gc] : 0.0;
        }
    };

    int write_idx = 0;
    load_tiles(0, write_idx);
    __syncthreads();
    
    for (int k = 0; k < K; k += BK) {
        int read_idx = write_idx;
        write_idx = 1 - write_idx;
        if (k + BK < K) load_tiles(k + BK, write_idx);
        
        // Compute Loop over BK
        // Step size DMMA_K = 4.
        // BK = 16. So 4 steps.
        #pragma unroll
        for (int ks = 0; ks < BK; ks += 4) {
            
            // Load A frags for WarpRows
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                wmma::load_matrix_sync(a_frags[i], &As[read_idx][warp_row * WM + i * 8][ks], BK + PAD);
            }
            
            // Load B frags for WarpCols
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                wmma::load_matrix_sync(b_frags[j], &Bs[read_idx][ks][warp_col * WN + j * 8], BN + PAD);
            }
            
            // Math
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    wmma::mma_sync(acc[i][j], a_frags[i], b_frags[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }
    
    // Store
    // Reusing As: 
    // We need 8 warps * 16x32 * 8 bytes = 4KB * 8 = 32KB.
    // As capacity: 2 * 64 * 20 * 8 = 20KB.
    // NOT ENOUGH Space to store all Warp results at once in SMEM.
    // But we don't need to store all at once.
    // We can store Tile-by-Tile? 
    // Or just increase SMEM size if hardware allows.
    // 32KB is fine for most GPUs.
    // Let's rely on standard store_matrix_sync to global if needed? NO, need beta.
    // Let's implement serialization via smaller buffer if needed, OR:
    // Redefine shared memory to be `union` of (As+Bs) and (OutputBuffer).
    // As+Bs = 20KB + 20KB = 40KB.
    // Output = 32KB. 
    // It fits!
    
    double* sm_out = reinterpret_cast<double*>(As);
    double* warp_sm = sm_out + warp_id * 512;
    
    // Store results to SMEM
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
             // Store 8x8 tile
             // Stride 32 (WN)
             wmma::store_matrix_sync(warp_sm + (i*8)*32 + (j*8), acc[i][j], 32, wmma::mem_row_major);
        }
    }
    __syncthreads();
    
    // Write out 16x32 tile to Global
    int gr_start = by*BM + warp_row*WM;
    int gc_start = bx*BN + warp_col*WN;
    
    // 512 elements. 32 threads. 16 elems/thread.
    for (int t = 0; t < 16; t++) {
        int tid_in_warp = tid % 32;
        int idx = tid_in_warp + t * 32; 
        
        int r = idx / 32;
        int c = idx % 32;
        
        int gr = gr_start + r;
        int gc = gc_start + c;
        
        if (gr < M && gc < N) {
            double val = warp_sm[idx];
            int addr = gr * ldc + gc;
            if (beta == 0.0) {
                C_batch[addr] = alpha * val;
            } else {
                C_batch[addr] = alpha * val + beta * C_batch[addr];
            }
        }
    }
}

extern "C" void mycublasDgemmStridedBatched(
    mycublasHandle_t handle,
    int M, int N, int K,
    const double alpha,
    const double *A, int lda, long long int strideA,
    const double *B, int ldb, long long int strideB,
    const double beta,
    double *C, int ldc, long long int strideC,
    int batchCount)
{
    cudaStream_t stream = (handle != nullptr) ? handle->stream : 0;
    
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, batchCount);
    dim3 block(256);
    
    // As+Bs size = 40KB roughly.
    // Check occupancy.
    
    dgemm_strided_batched_kernel<<<grid, block, 0, stream>>>(
        M, N, K, alpha,
        A, lda, strideA,
        B, ldb, strideB,
        beta, C, ldc, strideC
    );
}
