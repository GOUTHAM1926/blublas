#include "mycublas.h"
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

// ============================================================================
// FP64 WMMA/DMMA KERNEL
// Tile: 64x64x8
// Threads: 256
// ============================================================================

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 8; // Small K due to double precision DMMA constraints

constexpr int WM = 16;
constexpr int WN = 32;
constexpr int DMMA_M = 8;
constexpr int DMMA_N = 8;
constexpr int DMMA_K = 4;
constexpr int PAD = 4; // Double alignment

__global__ void dgemm_optimized_kernel(
    int M, int N, int K,
    double alpha,
    const double* __restrict__ A, int lda,
    const double* __restrict__ B, int ldb,
    double beta,
    double* __restrict__ C, int ldc)
{
    // Indexes
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int warp_row = warp_id / 2; // 0..3 (4 rows of warps)
    const int warp_col = warp_id % 2; // 0..1 (2 cols of warps)
    
    // Pointers
    // A: Row Major.
    const double *Ap = A;
    const double *Bp = B;
    double *Cp = C;
    
    // Shared Memory
    // As[2][BM][BK + PAD]
    // BM=64, BK=16 (needs to be multiple of DMMA_K=4. Larger BK better for reuse).
    // Let's try BK=16. 
    // Size: 2 * 64 * 16 * 8 bytes = 16KB. Very small. OK.
    const int MY_BK = 16;
    
    __shared__ double As[2][BM][MY_BK + PAD];
    __shared__ double Bs[2][MY_BK][BN + PAD];
    
    // Fragments
    // WarpTile 16x32.
    // DMMA 8x8x4.
    // M dim: 16 / 8 = 2.
    // N dim: 32 / 8 = 4.
    // Total 2x4 = 8 mma ops.
    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> af[2][4]; // dim K (buffer depth for loop)? No, tiling spatial.
    // Wait, matrix_a fragment matches mma_sync shape. 
    // We need 'a' fragments for M dimension (2 of them).
    // range i: 0..1.
    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> a_frags[2];

    // we need 'b' fragments for N dimension (4 of them).
    // range j: 0..3.
    wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> b_frags[4];

    // accumulators [2][4]
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
        for (int i = tid; i < BM * MY_BK; i += 256) {
            int r = i / MY_BK;
            int c = i % MY_BK;
            int gr = by * BM + r;
            int gc = ko + c;
            As[idx][r][c] = (gr < M && gc < K) ? Ap[gr*lda + gc] : 0.0;
        }
        // Load B: BK x BN (16 x 64). 1024 elements.
        for (int i = tid; i < MY_BK * BN; i += 256) {
            int r = i / BN;
            int c = i % BN;
            int gr = ko + r;
            int gc = bx * BN + c;
            Bs[idx][r][c] = (gr < K && gc < N) ? Bp[gr*ldb + gc] : 0.0;
        }
    };

    int write_idx = 0;
    load_tiles(0, write_idx);
    __syncthreads();
    
    for (int k = 0; k < K; k += MY_BK) {
        int read_idx = write_idx;
        write_idx = 1 - write_idx;
        if (k + MY_BK < K) load_tiles(k + MY_BK, write_idx);
        
        // Compute Loop over BK
        // Step size DMMA_K = 4.
        // MY_BK = 16. So 4 steps.
        #pragma unroll
        for (int ks = 0; ks < MY_BK; ks += 4) {
            
            // Load A frags for WarpRows
            // Warp covers rows: warp_row * WM (16) ... +15.
            // i=0 -> rows 0..7 local. i=1 -> rows 8..15 local.
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                wmma::load_matrix_sync(a_frags[i], &As[read_idx][warp_row * WM + i * 8][ks], MY_BK + PAD);
            }
            
            // Load B frags for WarpCols
            // Warp covers cols: warp_col * WN (32) ... +31.
            // j=0..3 -> cols 0,8,16,24 offsets.
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
    
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {

            int r = by*BM + warp_row*WM + i*8;
            int c = bx*BN + warp_col*WN + j*8;
            
            if (r < M && c < N) {
                
                double* sm = reinterpret_cast<double*>(As);
        
            }
        }
    }
    
    double* sm_out = reinterpret_cast<double*>(As);
    double* warp_sm = sm_out + warp_id * 512;
    
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
           
             wmma::store_matrix_sync(warp_sm + (i*8)*32 + (j*8), acc[i][j], 32, wmma::mem_row_major);
        }
    }
    __syncthreads();
    // No syncthreads needed between warps if disjoint regions.
    
    // Now write out warp_sm (16x32) to Global
    // 512 elements. 32 threads. 16 elems/thread.
    int gr_start = by*BM + warp_row*WM;
    int gc_start = bx*BN + warp_col*WN;
    
    for (int t = 0; t < 16; t++) {
        int tid_in_warp = tid % 32;
        int idx = tid_in_warp + t * 32; // 0..511
        // Map idx to row/col in 16x32 tile
        int r = idx / 32;
        int c = idx % 32;
        
        int gr = gr_start + r;
        int gc = gc_start + c;
        
        if (gr < M && gc < N) {
            double val = warp_sm[idx];
            double c_val = Cp[gr * ldc + gc];
            Cp[gr * ldc + gc] = alpha * val + beta * c_val;
        }
    }
}

extern "C" void mycublasDgemm(
    mycublasHandle_t handle,
    mycublasOperation_t transa, mycublasOperation_t transb,
    int M, int N, int K,
    double alpha,
    const double *A, int lda,
    const double *B, int ldb,
    double beta,
    double *C, int ldc)
{
    cudaStream_t stream = (handle != nullptr) ? handle->stream : 0;
    dim3 block(256);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dgemm_optimized_kernel<<<grid, block, 0, stream>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
