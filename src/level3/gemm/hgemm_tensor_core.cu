#include "mycublas.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// ============================================================================
// FP16 WMMA KERNEL (17 TFLOPS Version)
// BM=128, BN=128, BK=32, Threads=512
// ============================================================================

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int WM = 32; // Warp M
constexpr int WN = 32; // Warp N
constexpr int PAD = 8;

__global__ void hgemm_optimized_kernel(
    int M, int N, int K,
    __half alpha,
    const __half* __restrict__ A, int lda,
    const __half* __restrict__ B, int ldb,
    __half beta,
    __half* __restrict__ C, int ldc)
{
    // Block Index
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // Warp & Thread Index
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    // 512 threads = 16 warps.
    // We map warps to a 4x4 grid roughly? Or 4xSomething.
    // Block size 128x128.
    // If Warp tile is 32x32.
    // (128/32) * (128/32) = 4 * 4 = 16 warps.
    // Perfect match. 
    const int warp_row = warp_id / 4; // 0..3
    const int warp_col = warp_id % 4; // 0..3
    
    // Pointers
    const __half *Ap = A; // Will offset later
    const __half *Bp = B; // Will offset later
    __half *Cp = C;       // Will offset later

    // Shared Memory Double Buffering
    // As[2][BM][BK+PAD]
    // 2 * 128 * 40 * 2 bytes = ~20KB. OK.
    __shared__ __half As[2][BM][BK + PAD];
    __shared__ __half Bs[2][BK][BN + PAD];
   
    // WMMA Fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af[2]; // 2 fragments along M? 
    // Wait, WarpTile 32x32. WMMA is 16x16.
    // We need 2x2 WMMA operations per WarpTile.
    // Actually, let's look at reference loop.
    // for i=0..2, j=0..2: wmma::mma_sync...
    // So 2x2 grid of 16x16 ops per warp. Correct.
    
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bf[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> acc[2][2];

    // Initialize accumulators
    #pragma unroll
    for(int i=0; i<2; i++) 
        for(int j=0; j<2; j++) 
            wmma::fill_fragment(acc[i][j], __float2half(0.0f));

    // Lambda for loading to Shared Memory
    auto load_tiles = [&](int ko, int idx) {
        // Load A: BM x BK (128 x 32). Total 4096 elements.
        // Threads: 512. Each thread loads 8 elements.
        // Use vectorized load float4 (8 halves) if possible?
        // Let's stick to simple loop for correctness first or semi-vectorized.
        // 512 threads -> stride 512.
        for (int i = tid; i < BM * BK; i += 512) {
            int r = i / BK;
            int c = i % BK;
            // Global A: Row Major. A[by*BM + r][ko + c]
            int global_r = by * BM + r;
            int global_c = ko + c;
            
            __half val = __float2half(0.0f);
            if (global_r < M && global_c < K) {
                val = Ap[global_r * lda + global_c];
            }
            As[idx][r][c] = val;
        }

        // Load B: BK x BN (32 x 128). Total 4096 elements.
        for (int i = tid; i < BK * BN; i += 512) {
            int r = i / BN;
            int c = i % BN;
            // Global B: Row Major. B[ko + r][bx*BN + c]
            int global_r = ko + r;
            int global_c = bx * BN + c;
            
            __half val = __float2half(0.0f);
            if (global_r < K && global_c < N) {
                val = Bp[global_r * ldb + global_c];
            }
            Bs[idx][r][c] = val;
        }
    };

    // Prologue
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

        // Compute on read_idx
        // Loop over K-step inside tile (BK=32). 
        // WMMA K=16. So 2 steps.
        #pragma unroll
        for (int ks = 0; ks < BK; ks += 16) {
            
            // Load A fragments for the warp
            // Warp covers 32 rows of A (starting at warp_row * 32).
            // We need 2 fragments of 16 rows.
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                // As is row-major. Addr: As[read_idx][warp_row*32 + i*16][ks]
                // Note: &As... needs pointer to __half. Cast to const __half* for load_matrix_sync if needed?
                // Shared mem pointer is fine.
                wmma::load_matrix_sync(af[i], &As[read_idx][warp_row * WM + i * 16][ks], BK + PAD);
                
                // Load B fragments for the warp
                // Warp covers 32 cols of B (starting at warp_col * 32).
                // We need 2 fragments of 16 cols.
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                     if (i == 0) { // reuse bf loads across i iteration? No, bf depends on j.
                         // But we can reload bf[j] inside j loop.
                         // Optimization: move bf load out of i loop?
                         // Ops: C[i][j] += A[i] * B[j].
                         // A[i] is constant for all j. B[j] is constant for all i.
                         // Standard loop:
                         // for i: load A[i]
                         //   for j: load B[j], mma
                         // Better: load all B potentially? Registers limited.
                         // Reference does:
                         /*
                            for i=0..2: load A
                            for j=0..2: load B, mma
                         */
                         // This reloads B for each i.
                         wmma::load_matrix_sync(bf[j], &Bs[read_idx][ks][warp_col * WN + j * 16], BN + PAD);
                     }
                     wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

    // Epilogue: Store to Shared then Global?
    // We can store directly if swizzle compliant, but generic store is safer via shared.
    // Cast As to __half* for ease of pointer arithmetic?
    // Reference uses As buffer to store output results from registers before writing to global.
    // Reusing As: "sm = reinterpret_cast<__half*>(As)"
    // 256 elements per Warp (32x32 results). 16 Warps. Total 4096 elements.
    // As capacity: 2 * 128 * 40 = 10240 elements. Plenty of space.
    // Warp ID 0..15.
    // Offset each warp by say 256 elements? 
    // 16 * 256 = 4096. Fits.
    
    __half* sm = reinterpret_cast<__half*>(As);
    __half* warp_sm = sm + warp_id * 256; 

    // Store Accumulators to Shared
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            // Store 16x16 tile to shared memory
            // We need to map this 16x16 tile to legal row-major layout in SMEM
            // so we can coalesce write to global.
            // WarpTile is 32x32.
            // Sub-tile (i,j) is at offset (i*16, j*16) within WarpTile.
            // Stride in SMEM for this warp tile should be 32 (WN).
            // But wmma::store_matrix_sync expects a pointer and stride.
            // Let's interpret 'warp_sm' as a generic pointer to 32x32 block.
            // row stride = 16? No, 32.
            
            // Addr in warp_sm: i*16 * 32 + j*16
            // stride: 32 cannot be hardcoded if we want linear memory?
            // Actually, we can define stride=16. 
            // Better: Store each 16x16 block linearly or tiled?
            // Reference stores with stride 16 ("wmma::mem_row_major").
            // "wmma::store_matrix_sync(wsm, acc[i][j], 16, wmma::mem_row_major);"
            // This suggests wsm points to a 16x16 block, not part of 32x32.
            // Reference has: "float* wsm = sm + warp_id * 256;"
            // It reuses the SAME SMEM buffer sequentially!
            // Wait, "wsm" in reference seems to be overwritten?
            // Ah, reference computes indices carefully.
            /*
              for i..2, j..2:
                 store to wsm (size 256? no 16x16=256)
                 Loop 16x16 to write to device.
            */
            // So it serializes the write back to global memory via a small buffer?
            // wsm = sm + warp_id * 256. 
            // Total SMEM needed = 16 warps * 256 = 4096 elements.
            // As array is big enough.
            // This is smart. It avoids complex striding logic for the full tile.
            
            wmma::store_matrix_sync(warp_sm, acc[i][j], 16, wmma::mem_row_major);
            
            // Manual coalesced write loop (mostly)
            // 16x16 = 256 elements.
            // Warp has 32 threads.
            // Each thread writes 256 / 32 = 8 elements.
            // This is efficient.
            
            // Global offsets for this sub-tile
            int global_tile_r = by * BM + warp_row * WM + i * 16;
            int global_tile_c = bx * BN + warp_col * WN + j * 16;
            
            for (int t = 0; t < 8; t++) {
                // Thread mapping within 16x16 tile
                // 32 threads. elements 0..255.
                // ele = tid_in_warp + t * 32.
                int tid_in_warp = tid % 32;
                int ele_idx = tid_in_warp + t * 32;
                
                int r = ele_idx / 16; // 0..15
                int c = ele_idx % 16; // 0..15
                
                if (global_tile_r + r < M && global_tile_c + c < N) {
                   __half val = warp_sm[ele_idx];
                   int addr = (global_tile_r + r) * ldc + (global_tile_c + c);
                   
                   if (beta == __float2half(0.0f)) {
                       Cp[addr] = alpha * val;
                   } else {
                       Cp[addr] = alpha * val + beta * Cp[addr];
                   }
                }
            }
        }
    }
}

extern "C" void mycublasHgemmStridedBatched_dispatcher(mycublasHandle_t handle, mycublasOperation_t transa, mycublasOperation_t transb, int M, int N, int K, const __half alpha, const __half* A, int lda, long long sA, const __half* B, int ldb, long long sB, const __half beta, __half* C, int ldc, long long sC, int batchCount);

extern "C" void mycublasHgemm(
    mycublasHandle_t handle,
    mycublasOperation_t transa, mycublasOperation_t transb,
    int M, int N, int K,
    const __half alpha,
    const __half *d_A, int lda,
    const __half *d_B, int ldb,
    const __half beta,
    __half *d_C, int ldc)
{
    // Preferred path: Ada/Ampere optimized dispatcher
    if (handle && (handle->is_sm89 || handle->sm_count > 0)) {
        mycublasHgemmStridedBatched_dispatcher(handle, transa, transb, M, N, K, alpha, d_A, lda, 0, d_B, ldb, 0, beta, d_C, ldc, 0, 1);
        return;
    }

    // Fallback: Original WMMA kernel
    cudaStream_t stream = (handle != nullptr) ? handle->stream : 0;
    dim3 block(512);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    hgemm_optimized_kernel<<<grid, block, 0, stream>>>(M, N, K, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
}
