
// #pragma once

// #include <cuda_runtime.h>
// #include <stdint.h>
// #include <stdio.h>

// #ifndef MMA_TF32
// #define MMA_TF32(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1)                \
//     asm volatile(                                                   \
//         "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "     \
//         "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"       \
//         : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)                     \
//         : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1))
// #endif

// // Tile Configuration struct that automatically deduces load patterns
// template <int BM_, int BN_, int BK_, int STAGES_, int THREADS_>
// struct SgemmTileConfig {
//     static constexpr int BM = BM_;
//     static constexpr int BN = BN_;
//     static constexpr int BK = BK_;
//     static constexpr int STAGES = STAGES_;
//     static constexpr int THREADS = THREADS_;

//     static constexpr int AS_SIZE = BM * BK;
//     static constexpr int BS_SIZE = BN * BK;
//     static constexpr int STAGE_SIZE = AS_SIZE + BS_SIZE;

//     // ----------------------------------------------------------------
//     // NT Normal Load Config (Contiguous in K)
//     // A: [M, K], contiguous along K. B: [N, K], contiguous along K.
//     // ----------------------------------------------------------------
//     static constexpr int NT_VEC_A = (BK % 4 == 0) ? 4 : 1; 
//     static constexpr int NT_THREADS_PER_ROW_A = BK / NT_VEC_A;
//     static constexpr int NT_ROWS_PER_ITER_A = THREADS / NT_THREADS_PER_ROW_A;
//     static constexpr int NT_LOAD_ITERS_A = BM / NT_ROWS_PER_ITER_A;

//     static constexpr int NT_VEC_B = (BK % 4 == 0) ? 4 : 1;
//     static constexpr int NT_THREADS_PER_ROW_B = BK / NT_VEC_B;
//     static constexpr int NT_ROWS_PER_ITER_B = THREADS / NT_THREADS_PER_ROW_B;
//     static constexpr int NT_LOAD_ITERS_B = BN / NT_ROWS_PER_ITER_B;

//     // ----------------------------------------------------------------
//     // TN Normal Load Config (Contiguous in M/N)
//     // A: [K, M], contiguous along M. B: [K, N], contiguous along N.
//     // ----------------------------------------------------------------
//     static constexpr int TN_VEC_A = (BM % 4 == 0) ? 4 : 1;
//     static constexpr int TN_THREADS_PER_ROW_A = BM / TN_VEC_A;
//     static constexpr int TN_ROWS_PER_ITER_A = THREADS / TN_THREADS_PER_ROW_A;
//     static constexpr int TN_LOAD_ITERS_A = BK / TN_ROWS_PER_ITER_A;

//     static constexpr int TN_VEC_B = (BN % 4 == 0) ? 4 : 1;
//     static constexpr int TN_THREADS_PER_ROW_B = BN / TN_VEC_B;
//     static constexpr int TN_ROWS_PER_ITER_B = THREADS / TN_THREADS_PER_ROW_B;
//     static constexpr int TN_LOAD_ITERS_B = BK / TN_ROWS_PER_ITER_B;

//     // ----------------------------------------------------------------
//     // Warp Layout Config
//     // ----------------------------------------------------------------
//     // Calculate the most square-like arrangement of warps that fits BM/BN ratio
//     static constexpr int WARP_COUNT = THREADS / 32;
//     // Wide tiles (BN > BM, e.g. 128x256 with 8 warps): need WARPS_M=2 not 1.
//     // Without this, 128x256 gets WARP_TILE=128x32, MMA_M=8, MMA_N=4 — imbalanced
//     // B-register preloading and 208 live registers vs the correct 192.
//     // Fix: for BN > BM, target square warp tiles by using WARP_COUNT/4 warps in M.
//     static constexpr int WARPS_M = (BM >= BN * 2)               ? (WARP_COUNT >= 4 ? 4 : 2) :
//                                    (BN > BM && WARP_COUNT >= 4)  ? (WARP_COUNT / 4)           :
//                                    (BM == BN && WARP_COUNT >= 4) ? 2                          :
//                                    (WARP_COUNT >= 2 ? 1 : 1);
//     static constexpr int WARPS_N = WARP_COUNT / WARPS_M;
    
//     static constexpr int WARP_TILE_M = BM / WARPS_M;
//     static constexpr int WARP_TILE_N = BN / WARPS_N;

//     static constexpr int MMA_M = WARP_TILE_M / 16;
//     static constexpr int MMA_N = WARP_TILE_N / 8;
// };

// // Mode Dispatchers
// enum class SgemmLayout {
//     NT,  // C = A * B^T
//     TN,  // C = A^T * B
//     NN   // C = A * B
// };

// template <typename Config, bool IsAligned, int SplitK, SgemmLayout Layout>
// __global__ void __launch_bounds__(Config::THREADS, 1)
// sgemm_backward_template_kernel(
//     int M, int N, int K,
//     float alpha,
//     const float* __restrict__ A, int lda, long long strideA,
//     const float* __restrict__ B, int ldb, long long strideB,
//     float beta,
//     float* __restrict__ C, int ldc, long long strideC,
//     int batchCount)
// {
//     const int batch = blockIdx.z / SplitK;
//     const int sk_idx = blockIdx.z % SplitK;
//     if (batch >= batchCount) return;

//     // CTA Swizzling (16-wide)
//     const int grid_x = (N + Config::BN - 1) / Config::BN;
//     const int grid_y = (M + Config::BM - 1) / Config::BM;
//     const int block_id = blockIdx.y * grid_x + blockIdx.x;
//     const int sw = max(1, min(grid_y, 16));
//     const int bx = (block_id / sw) % grid_x;
//     const int by = (block_id % sw) + (block_id / (grid_x * sw)) * sw;

//     if (by * Config::BM >= M || bx * Config::BN >= N) return;

//     const int tid  = (int)threadIdx.x;
//     const int lane = tid & 31, wid = tid >> 5;
//     const int wy = wid / Config::WARPS_N, wx = wid % Config::WARPS_N;

//     const int k_tiles = (K + Config::BK - 1) / Config::BK;
//     const int tiles_per_sk = (k_tiles + SplitK - 1) / SplitK;
//     const int k_start = sk_idx * tiles_per_sk * Config::BK;
//     const int k_end = min(K, (sk_idx + 1) * tiles_per_sk * Config::BK);

//     extern __shared__ float smem[];
//     float acc[Config::MMA_M][Config::MMA_N][4];
//     #pragma unroll
//     for (int i = 0; i < Config::MMA_M; i++) 
//         #pragma unroll
//         for (int j = 0; j < Config::MMA_N; j++) 
//             #pragma unroll
//             for (int d = 0; d < 4; d++) acc[i][j][d] = 0.f;

//     // ---------------------------------------------------------
//     // Shared Memory Base Adjustments (Swizzle Masks)
//     // ---------------------------------------------------------
//     // Depends on BK to avoid bank conflicts. For BK=16, mask = >> 2
//     // For BK=32, mask = >> 3
//     constexpr int SMEM_MASK_A = (Config::BK == 16) ? ((Config::BM >= 64 ? 3 : 1) << 2) : (7 << 3);
//     constexpr int SMEM_MASK_B = (Config::BK == 16) ? ((Config::BN >= 64 ? 3 : 1) << 2) : (7 << 3);

//     // ---------------------------------------------------------
//     // Global Pointers
//     // ---------------------------------------------------------
//     const float* gA_ptr[Layout == SgemmLayout::TN ? Config::TN_LOAD_ITERS_A : Config::NT_LOAD_ITERS_A];
//     const float* gB_ptr[Layout == SgemmLayout::NT ? Config::NT_LOAD_ITERS_B : Config::TN_LOAD_ITERS_B];

//     if constexpr (Layout == SgemmLayout::NT || Layout == SgemmLayout::NN) {
//         #pragma unroll
//         for (int i = 0; i < Config::NT_LOAD_ITERS_A; i++) {
//             int r = (tid / Config::NT_THREADS_PER_ROW_A) + i * Config::NT_ROWS_PER_ITER_A;
//             int c = (tid % Config::NT_THREADS_PER_ROW_A) * 4;
//             gA_ptr[i] = A + (long long)batch * strideA + (long long)(by * Config::BM + r) * lda + (k_start + c);
//         }
//     } else { // TN
//         #pragma unroll
//         for (int i = 0; i < Config::TN_LOAD_ITERS_A; i++) {
//             int r = (tid / Config::TN_THREADS_PER_ROW_A) + i * Config::TN_ROWS_PER_ITER_A; // This is K
//             int c = (tid % Config::TN_THREADS_PER_ROW_A) * 4; // This is M
//             gA_ptr[i] = A + (long long)batch * strideA + (long long)(k_start + r) * lda + (by * Config::BM + c);
//         }
//     }

//     if constexpr (Layout == SgemmLayout::NT) {
//         #pragma unroll
//         for (int i = 0; i < Config::NT_LOAD_ITERS_B; i++) {
//             int r = (tid / Config::NT_THREADS_PER_ROW_B) + i * Config::NT_ROWS_PER_ITER_B;
//             int c = (tid % Config::NT_THREADS_PER_ROW_B) * 4;
//             gB_ptr[i] = B + (long long)batch * strideB + (long long)(bx * Config::BN + r) * ldb + (k_start + c);
//         }
//     } else { // TN or NN
//         #pragma unroll
//         for (int i = 0; i < Config::TN_LOAD_ITERS_B; i++) {
//             int r = (tid / Config::TN_THREADS_PER_ROW_B) + i * Config::TN_ROWS_PER_ITER_B; // This is K
//             int c = (tid % Config::TN_THREADS_PER_ROW_B) * 4; // This is N
//             gB_ptr[i] = B + (long long)batch * strideB + (long long)(k_start + r) * ldb + (bx * Config::BN + c);
//         }
//     }

//     auto load_to_stage = [&](int stage, int ko) {
//         float* As = smem + stage * Config::STAGE_SIZE;
//         float* Bs = As + Config::AS_SIZE;
        
//         if constexpr (Layout == SgemmLayout::NT || Layout == SgemmLayout::NN) {
//             #pragma unroll
//             for (int i = 0; i < Config::NT_LOAD_ITERS_A; i++) {
//                 int r = (tid / Config::NT_THREADS_PER_ROW_A) + i * Config::NT_ROWS_PER_ITER_A;
//                 int c = (tid % Config::NT_THREADS_PER_ROW_A) * 4;
//                 int sc = c ^ (((r & 7) << 2) & (Config::BK - 1)); 

//                 uint32_t sm_a = __cvta_generic_to_shared(&As[r * Config::BK + sc]);
//                 int gr = by * Config::BM + r, gc = ko + c;
//                 if constexpr (IsAligned) {
//                     int bytes = (gr < M && gc < K) ? max(0, min(16, (K - gc) * 4)) : 0;
//                     asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm_a), "l"((const void*)gA_ptr[i]), "r"(bytes));
//                 } else {
//                     float4 val = {0,0,0,0};
//                     if (gr < M && gc < K) {
//                         val.x = gA_ptr[i][0]; if (gc+1 < K) val.y = gA_ptr[i][1];
//                         if (gc+2 < K) val.z = gA_ptr[i][2]; if (gc+3 < K) val.w = gA_ptr[i][3];
//                     }
//                     *(float4*)&As[r * Config::BK + sc] = val;
//                 }
//                 gA_ptr[i] += Config::BK;
//             }
//         } else { // TN
//             #pragma unroll
//             for (int i = 0; i < Config::TN_LOAD_ITERS_A; i++) {
//                 int r = (tid / Config::TN_THREADS_PER_ROW_A) + i * Config::TN_ROWS_PER_ITER_A; // K
//                 int c = (tid % Config::TN_THREADS_PER_ROW_A) * 4; // M
//                 int sc = c ^ ((r & 7) << 3); // Swizzle along M dimension for 128
                
//                 uint32_t sm_a = __cvta_generic_to_shared(&As[r * Config::BM + sc]);
//                 int gk = ko + r, gm = by * Config::BM + c;
//                 if constexpr (IsAligned) {
//                     int bytes = (gk < K && gm < M) ? max(0, min(16, (M - gm) * 4)) : 0;
//                     asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm_a), "l"((const void*)gA_ptr[i]), "r"(bytes));
//                 } else {
//                     float4 val = {0,0,0,0};
//                     if (gk < K) {
//                         if (gm < M) val.x = gA_ptr[i][0];
//                         if (gm + 1 < M) val.y = gA_ptr[i][1];
//                         if (gm + 2 < M) val.z = gA_ptr[i][2];
//                         if (gm + 3 < M) val.w = gA_ptr[i][3];
//                     }
//                     *(float4*)&As[r * Config::BM + sc] = val;
//                 }
//                 gA_ptr[i] += Config::BK * lda;
//             }
//         }

//         if constexpr (Layout == SgemmLayout::NT) {
//             #pragma unroll
//             for (int i = 0; i < Config::NT_LOAD_ITERS_B; i++) {
//                 int r = (tid / Config::NT_THREADS_PER_ROW_B) + i * Config::NT_ROWS_PER_ITER_B;
//                 int c = (tid % Config::NT_THREADS_PER_ROW_B) * 4;
//                 int sc = c ^ (((r & 7) << 2) & (Config::BK - 1));

//                 uint32_t sm_b = __cvta_generic_to_shared(&Bs[r * Config::BK + sc]);
//                 int gr = bx * Config::BN + r, gc = ko + c;
//                 if constexpr (IsAligned) {
//                     int bytes = (gr < N && gc < K) ? max(0, min(16, (K - gc) * 4)) : 0;
//                     asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm_b), "l"((const void*)gB_ptr[i]), "r"(bytes));
//                 } else {
//                     float4 val = {0,0,0,0};
//                     if (gr < N && gc < K) {
//                         val.x = gB_ptr[i][0]; if (gc+1 < K) val.y = gB_ptr[i][1];
//                         if (gc+2 < K) val.z = gB_ptr[i][2]; if (gc+3 < K) val.w = gB_ptr[i][3];
//                     }
//                     *(float4*)&Bs[r * Config::BK + sc] = val;
//                 }
//                 gB_ptr[i] += Config::BK;
//             }
//         } else { // TN or NN
//             #pragma unroll
//             for (int i = 0; i < Config::TN_LOAD_ITERS_B; i++) {
//                 int r = (tid / Config::TN_THREADS_PER_ROW_B) + i * Config::TN_ROWS_PER_ITER_B; // K
//                 int c = (tid % Config::TN_THREADS_PER_ROW_B) * 4; // N
//                 int sc = c ^ (((r & 7) << 2) & (Config::BN - 1));
                
//                 uint32_t sm_b = __cvta_generic_to_shared(&Bs[r * Config::BN + sc]);
//                 int gk = ko + r, gn = bx * Config::BN + c;
//                 if constexpr (IsAligned) {
//                     int bytes = (gk < K && gn < N) ? max(0, min(16, (N - gn) * 4)) : 0;
//                     asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm_b), "l"((const void*)gB_ptr[i]), "r"(bytes));
//                 } else {
//                     float4 val = {0,0,0,0};
//                     if (gk < K) {
//                         if (gn < N) val.x = gB_ptr[i][0];
//                         if (gn + 1 < N) val.y = gB_ptr[i][1];
//                         if (gn + 2 < N) val.z = gB_ptr[i][2];
//                         if (gn + 3 < N) val.w = gB_ptr[i][3];
//                     }
//                     *(float4*)&Bs[r * Config::BN + sc] = val;
//                 }
//                 gB_ptr[i] += Config::BK * ldb;
//             }
//         }
//     };

//     // -------------------------------------------------------------
//     // WMMA Loads
//     // -------------------------------------------------------------
//     const int g_sh = lane / 4, t_sh = lane % 4;
    
//     // Y-axis wmma pointers
//     int rbaseA[Config::MMA_M], maskA[Config::MMA_M];
//     #pragma unroll
//     for (int i = 0; i < Config::MMA_M; i++) {
//         rbaseA[i] = (wy * Config::WARP_TILE_M + i * 16 + g_sh) * ((Layout == SgemmLayout::NT || Layout == SgemmLayout::NN) ? Config::BK : 1);
//         if constexpr (Layout == SgemmLayout::NT || Layout == SgemmLayout::NN) {
//             maskA[i] = (((wy * Config::WARP_TILE_M + i * 16 + g_sh) & 7) << 2) & (Config::BK - 1);
//         } else { // TN
//             maskA[i] = (((wy * Config::WARP_TILE_M + i * 16 + g_sh) & 7) << 3) & (Config::BM - 1);
//         }
//     }
    
//     // X-axis wmma pointers
//     int rbaseB[Config::MMA_N];
//     #pragma unroll
//     for (int j = 0; j < Config::MMA_N; j++) rbaseB[j] = (wx * Config::WARP_TILE_N + j * 8 + g_sh) * ((Layout == SgemmLayout::NT) ? Config::BK : 1);

//     auto load_frA = [&](uint32_t reg[4], int ks, int mi, int st) {
//         float* As = smem + st * Config::STAGE_SIZE;
//         if constexpr (Layout == SgemmLayout::NT || Layout == SgemmLayout::NN) {
//             auto ga = [&](int row_idx, int c) { return *(uint32_t*)&As[row_idx + (c ^ maskA[mi])]; };
//             reg[0] = ga(rbaseA[mi], ks + t_sh);
//             reg[1] = ga(rbaseA[mi] + 8 * Config::BK, ks + t_sh);
//             reg[2] = ga(rbaseA[mi], ks + t_sh + 4);
//             reg[3] = ga(rbaseA[mi] + 8 * Config::BK, ks + t_sh + 4);
//         } else {
//             const int k0 = ks + t_sh, k4 = k0 + 4, row = (wy * Config::WARP_TILE_M + mi * 16 + g_sh);
//             auto ga = [&](int k, int m) { return *(uint32_t*)&As[k * Config::BM + (m ^ ((k & 7) << 3))]; };
//             reg[0] = ga(k0, row);
//             reg[1] = ga(k0, row + 8);
//             reg[2] = ga(k4, row);
//             reg[3] = ga(k4, row + 8);
//         }
//     };

//     auto load_frB = [&](uint32_t reg[2], int ks, int ni, int st) {
//         float* Bs = smem + st * Config::STAGE_SIZE + Config::AS_SIZE;
//         if constexpr (Layout == SgemmLayout::NT) {
//             const int row = rbaseB[ni], mask = (((row / Config::BK) & 7) << 2) & (Config::BK - 1);
//             auto gb = [&](int r, int c) { return *(uint32_t*)&Bs[r + (c ^ mask)]; };
//             reg[0] = gb(row, ks + t_sh); reg[1] = gb(row, ks + t_sh + 4);
//         } else {
//             const int k0 = ks + t_sh, k4 = k0 + 4, col = (wx * Config::WARP_TILE_N + ni * 8 + g_sh);
//             auto gb = [&](int k, int n) { return *(uint32_t*)&Bs[k * Config::BN + (n ^ ((k & 7) << 2))]; };
//             reg[0] = gb(k0, col); reg[1] = gb(k4, col);
//         }
//     };

//     if (k_start < k_end) {
//         load_to_stage(0, k_start); asm volatile("cp.async.commit_group;\n");
//         #pragma unroll
//         for (int s = 1; s < Config::STAGES - 1; s++) {
//             if (k_start + s * Config::BK < k_end) load_to_stage(s, k_start + s * Config::BK); 
//             asm volatile("cp.async.commit_group;\n");
//         }
//     }

//     int ws = Config::STAGES - 1, rs = 0; 
//     uint32_t frA[2][Config::MMA_M][4], frB[2][Config::MMA_N][2];
    
//     asm volatile("cp.async.wait_group %0;\n" :: "n"(Config::STAGES - 2)); __syncthreads();
//     #pragma unroll
//     for (int i = 0; i < Config::MMA_M; i++) load_frA(frA[0][i], 0, i, rs);
//     #pragma unroll
//     for (int j = 0; j < Config::MMA_N; j++) load_frB(frB[0][j], 0, j, rs);

//     for (int k = k_start; k < k_end; k += Config::BK) {
//         if (k + (Config::STAGES - 1) * Config::BK < k_end) load_to_stage(ws, k + (Config::STAGES - 1) * Config::BK); 
//         asm volatile("cp.async.commit_group;\n");
//         #pragma unroll
//         for (int ks = 0; ks < Config::BK; ks += 8) {
//             const int p = (ks / 8) & 1, q = p ^ 1;
//             #pragma unroll
//             for (int i = 0; i < Config::MMA_M; i++) {
//                 if (ks + 8 < Config::BK) { 
//                     load_frA(frA[q][i], ks + 8, i, rs); 
//                     for (int j = i; j < Config::MMA_N; j += Config::MMA_M) load_frB(frB[q][j], ks + 8, j, rs);
//                 }
//                 else if (k + Config::BK < k_end) { 
//                     if (i == 0) { 
//                         asm volatile("cp.async.wait_group %0;\n" :: "n"(Config::STAGES - 2)); 
//                         __syncthreads(); 
//                         rs = (rs+1)%Config::STAGES; ws = (ws+1)%Config::STAGES; 
//                     }
//                     load_frA(frA[q][i], 0, i, rs); 
//                     for (int j = i; j < Config::MMA_N; j += Config::MMA_M) load_frB(frB[q][j], 0, j, rs);
//                 }
//                 #pragma unroll
//                 for (int j = 0; j < Config::MMA_N; j++) {
//                     // Safety check if MMA_M < MMA_N/2 (e.g. 64x128 where MMA_M=4, MMA_N=16)
//                     // We must load all B regs even if i exhausted.
//                     if (ks + 8 < Config::BK && i == Config::MMA_M - 1 && Config::MMA_N/2 > Config::MMA_M) {
//                         #pragma unroll
//                         for(int rB = Config::MMA_M*2; rB < Config::MMA_N; rB++) load_frB(frB[q][rB], ks+8, rB, rs);
//                     }
//                     else if (k + Config::BK < k_end && i == Config::MMA_M - 1 && Config::MMA_N/2 > Config::MMA_M) {
//                         #pragma unroll
//                         for(int rB = Config::MMA_M*2; rB < Config::MMA_N; rB++) load_frB(frB[q][rB], 0, rB, rs);
//                     }
                    
//                     MMA_TF32(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3], 
//                              frA[p][i][0], frA[p][i][1], frA[p][i][2], frA[p][i][3], 
//                              frB[p][j][0], frB[p][j][1]);
//                 }
//             }
//         }
//     }

//     // Epilogue: float4 stores where possible, else float2
//     // ----------------------------------------------------------------
//     // Epilogue: C[r,c] = alpha * acc + beta * dst
//     // ----------------------------------------------------------------
//     const int g_epi = lane / 4, t_epi = lane % 4;
//     float* dC_batch = C + (long long)batch * strideC;
//     const bool can_vectorize_c = IsAligned && ((ldc & 1) == 0) && (((size_t)dC_batch & 7) == 0);

//     #pragma unroll
//     for (int i = 0; i < Config::MMA_M; i++) {
//         const int r0 = by * Config::BM + wy * Config::WARP_TILE_M + i * 16 + g_epi;
//         const int r8 = r0 + 8;
//         if (r0 >= M) continue;

//         #pragma unroll
//         for (int j = 0; j < Config::MMA_N; j++) {
//             const int c0 = bx * Config::BN + wx * Config::WARP_TILE_N + j * 8 + t_epi * 2;
//             if (c0 >= N) continue;

//             auto store_row = [&](int r, int c, float f0, float f1) {
//                 if (r >= M) return;
//                 float* dst = &dC_batch[(long long)r * ldc + c];
//                 if constexpr (SplitK > 1) {
//                     atomicAdd(dst, alpha * f0);
//                     if (c + 1 < N) atomicAdd(dst + 1, alpha * f1);
//                 } else {
//                     if (can_vectorize_c && c + 1 < N) {
//                         float2 old = (beta == 0.f) ? make_float2(0,0) : *(float2*)dst;
//                         *(float2*)dst = { alpha * f0 + beta * old.x, alpha * f1 + beta * old.y };
//                     } else {
//                         dst[0] = alpha * f0 + (beta == 0 ? 0 : beta * dst[0]);
//                         if (c + 1 < N) dst[1] = alpha * f1 + (beta == 0 ? 0 : beta * dst[1]);
//                     }
//                 }
//             };

//             store_row(r0, c0, acc[i][j][0], acc[i][j][1]);
//             store_row(r8, c0, acc[i][j][2], acc[i][j][3]);
//         }
//     }

// }

// template <typename Config>
// __global__ void sgemm_scale_template_kernel(float* C, float beta, int M, int N, int ldc, long long strideC, int batchCount) {
//     int r = blockIdx.y * blockDim.y + threadIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x, b = blockIdx.z;
//     if (r < M && c < N && b < batchCount) {
//         float* dst = &C[b * strideC + (long long)r * ldc + c];
//         *dst = (beta == 0.f) ? 0.f : (*dst * beta);
//     }
// }



























#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#ifndef MMA_TF32
#define MMA_TF32(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1)                \
    asm volatile(                                                   \
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "     \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"       \
        : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)                     \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1))
#endif

// Tile Configuration struct that automatically deduces load patterns
template <int BM_, int BN_, int BK_, int STAGES_, int THREADS_>
struct SgemmTileConfig {
    static constexpr int BM = BM_;
    static constexpr int BN = BN_;
    static constexpr int BK = BK_;
    static constexpr int STAGES = STAGES_;
    static constexpr int THREADS = THREADS_;

    static constexpr int AS_SIZE = BM * BK;
    static constexpr int BS_SIZE = BN * BK;
    static constexpr int STAGE_SIZE = AS_SIZE + BS_SIZE;

    // ----------------------------------------------------------------
    // NT Normal Load Config (Contiguous in K)
    // A: [M, K], contiguous along K. B: [N, K], contiguous along K.
    // ----------------------------------------------------------------
    static constexpr int NT_VEC_A = (BK % 4 == 0) ? 4 : 1; 
    static constexpr int NT_THREADS_PER_ROW_A = BK / NT_VEC_A;
    static constexpr int NT_ROWS_PER_ITER_A = THREADS / NT_THREADS_PER_ROW_A;
    static constexpr int NT_LOAD_ITERS_A = BM / NT_ROWS_PER_ITER_A;

    static constexpr int NT_VEC_B = (BK % 4 == 0) ? 4 : 1;
    static constexpr int NT_THREADS_PER_ROW_B = BK / NT_VEC_B;
    static constexpr int NT_ROWS_PER_ITER_B = THREADS / NT_THREADS_PER_ROW_B;
    static constexpr int NT_LOAD_ITERS_B = BN / NT_ROWS_PER_ITER_B;

    // ----------------------------------------------------------------
    // TN Normal Load Config (Contiguous in M/N)
    // A: [K, M], contiguous along M. B: [K, N], contiguous along N.
    // ----------------------------------------------------------------
    static constexpr int TN_VEC_A = (BM % 4 == 0) ? 4 : 1;
    static constexpr int TN_THREADS_PER_ROW_A = BM / TN_VEC_A;
    static constexpr int TN_ROWS_PER_ITER_A = THREADS / TN_THREADS_PER_ROW_A;
    static constexpr int TN_LOAD_ITERS_A = BK / TN_ROWS_PER_ITER_A;

    static constexpr int TN_VEC_B = (BN % 4 == 0) ? 4 : 1;
    static constexpr int TN_THREADS_PER_ROW_B = BN / TN_VEC_B;
    static constexpr int TN_ROWS_PER_ITER_B = THREADS / TN_THREADS_PER_ROW_B;
    static constexpr int TN_LOAD_ITERS_B = BK / TN_ROWS_PER_ITER_B;

    // ----------------------------------------------------------------
    // Warp Layout Config
    // ----------------------------------------------------------------
    // Calculate the most square-like arrangement of warps that fits BM/BN ratio
    static constexpr int WARP_COUNT = THREADS / 32;
    static constexpr int WARPS_M = (BM >= BN * 2) ? (WARP_COUNT >= 4 ? 4 : 2) : 
                                   (BM == BN && WARP_COUNT >= 4 ? 2 : 
                                   (WARP_COUNT >= 2 ? 1 : 1));
    static constexpr int WARPS_N = WARP_COUNT / WARPS_M;
    
    static constexpr int WARP_TILE_M = BM / WARPS_M;
    static constexpr int WARP_TILE_N = BN / WARPS_N;

    static constexpr int MMA_M = WARP_TILE_M / 16;
    static constexpr int MMA_N = WARP_TILE_N / 8;
};

// Mode Dispatchers
enum class SgemmLayout {
    NT,  // C = A * B^T
    TN,  // C = A^T * B
    NN   // C = A * B
};

template <typename Config, bool IsAligned, int SplitK, SgemmLayout Layout>
__global__ void __launch_bounds__(Config::THREADS, 1)
sgemm_backward_template_kernel(
    int M, int N, int K,
    float alpha,
    const float* __restrict__ A, int lda, long long strideA,
    const float* __restrict__ B, int ldb, long long strideB,
    float beta,
    float* __restrict__ C, int ldc, long long strideC,
    int batchCount)
{
    const int batch = blockIdx.z / SplitK;
    const int sk_idx = blockIdx.z % SplitK;
    if (batch >= batchCount) return;

    // CTA Swizzling (16-wide)
    const int grid_x = (N + Config::BN - 1) / Config::BN;
    const int grid_y = (M + Config::BM - 1) / Config::BM;
    const int block_id = blockIdx.y * grid_x + blockIdx.x;
    const int sw = max(1, min(grid_y, 16));
    const int bx = (block_id / sw) % grid_x;
    const int by = (block_id % sw) + (block_id / (grid_x * sw)) * sw;

    if (by * Config::BM >= M || bx * Config::BN >= N) return;

    const int tid  = (int)threadIdx.x;
    const int lane = tid & 31, wid = tid >> 5;
    const int wy = wid / Config::WARPS_N, wx = wid % Config::WARPS_N;

    const int k_tiles = (K + Config::BK - 1) / Config::BK;
    const int tiles_per_sk = (k_tiles + SplitK - 1) / SplitK;
    const int k_start = sk_idx * tiles_per_sk * Config::BK;
    const int k_end = min(K, (sk_idx + 1) * tiles_per_sk * Config::BK);

    extern __shared__ float smem[];
    float acc[Config::MMA_M][Config::MMA_N][4];
    #pragma unroll
    for (int i = 0; i < Config::MMA_M; i++) 
        #pragma unroll
        for (int j = 0; j < Config::MMA_N; j++) 
            #pragma unroll
            for (int d = 0; d < 4; d++) acc[i][j][d] = 0.f;

    // ---------------------------------------------------------
    // Shared Memory Base Adjustments (Swizzle Masks)
    // ---------------------------------------------------------
    // Depends on BK to avoid bank conflicts. For BK=16, mask = >> 2
    // For BK=32, mask = >> 3
    constexpr int SMEM_MASK_A = (Config::BK == 16) ? ((Config::BM >= 64 ? 3 : 1) << 2) : (7 << 3);
    constexpr int SMEM_MASK_B = (Config::BK == 16) ? ((Config::BN >= 64 ? 3 : 1) << 2) : (7 << 3);

    // ---------------------------------------------------------
    // Global Pointers
    // ---------------------------------------------------------
    const float* gA_ptr[Layout == SgemmLayout::TN ? Config::TN_LOAD_ITERS_A : Config::NT_LOAD_ITERS_A];
    const float* gB_ptr[Layout == SgemmLayout::NT ? Config::NT_LOAD_ITERS_B : Config::TN_LOAD_ITERS_B];

    if constexpr (Layout == SgemmLayout::NT || Layout == SgemmLayout::NN) {
        #pragma unroll
        for (int i = 0; i < Config::NT_LOAD_ITERS_A; i++) {
            int r = (tid / Config::NT_THREADS_PER_ROW_A) + i * Config::NT_ROWS_PER_ITER_A;
            int c = (tid % Config::NT_THREADS_PER_ROW_A) * 4;
            gA_ptr[i] = A + (long long)batch * strideA + (long long)(by * Config::BM + r) * lda + (k_start + c);
        }
    } else { // TN
        #pragma unroll
        for (int i = 0; i < Config::TN_LOAD_ITERS_A; i++) {
            int r = (tid / Config::TN_THREADS_PER_ROW_A) + i * Config::TN_ROWS_PER_ITER_A; // This is K
            int c = (tid % Config::TN_THREADS_PER_ROW_A) * 4; // This is M
            gA_ptr[i] = A + (long long)batch * strideA + (long long)(k_start + r) * lda + (by * Config::BM + c);
        }
    }

    if constexpr (Layout == SgemmLayout::NT) {
        #pragma unroll
        for (int i = 0; i < Config::NT_LOAD_ITERS_B; i++) {
            int r = (tid / Config::NT_THREADS_PER_ROW_B) + i * Config::NT_ROWS_PER_ITER_B;
            int c = (tid % Config::NT_THREADS_PER_ROW_B) * 4;
            gB_ptr[i] = B + (long long)batch * strideB + (long long)(bx * Config::BN + r) * ldb + (k_start + c);
        }
    } else { // TN or NN
        #pragma unroll
        for (int i = 0; i < Config::TN_LOAD_ITERS_B; i++) {
            int r = (tid / Config::TN_THREADS_PER_ROW_B) + i * Config::TN_ROWS_PER_ITER_B; // This is K
            int c = (tid % Config::TN_THREADS_PER_ROW_B) * 4; // This is N
            gB_ptr[i] = B + (long long)batch * strideB + (long long)(k_start + r) * ldb + (bx * Config::BN + c);
        }
    }

    auto load_to_stage = [&](int stage, int ko) {
        float* As = smem + stage * Config::STAGE_SIZE;
        float* Bs = As + Config::AS_SIZE;
        
        if constexpr (Layout == SgemmLayout::NT || Layout == SgemmLayout::NN) {
            #pragma unroll
            for (int i = 0; i < Config::NT_LOAD_ITERS_A; i++) {
                int r = (tid / Config::NT_THREADS_PER_ROW_A) + i * Config::NT_ROWS_PER_ITER_A;
                int c = (tid % Config::NT_THREADS_PER_ROW_A) * 4;
                int sc = c ^ (((r & 7) << 2) & (Config::BK - 1)); 

                uint32_t sm_a = __cvta_generic_to_shared(&As[r * Config::BK + sc]);
                int gr = by * Config::BM + r, gc = ko + c;
                if constexpr (IsAligned) {
                    int bytes = (gr < M && gc < K) ? max(0, min(16, (K - gc) * 4)) : 0;
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm_a), "l"((const void*)gA_ptr[i]), "r"(bytes));
                } else {
                    float4 val = {0,0,0,0};
                    if (gr < M && gc < K) {
                        val.x = gA_ptr[i][0]; if (gc+1 < K) val.y = gA_ptr[i][1];
                        if (gc+2 < K) val.z = gA_ptr[i][2]; if (gc+3 < K) val.w = gA_ptr[i][3];
                    }
                    *(float4*)&As[r * Config::BK + sc] = val;
                }
                gA_ptr[i] += Config::BK;
            }
        } else { // TN
            #pragma unroll
            for (int i = 0; i < Config::TN_LOAD_ITERS_A; i++) {
                int r = (tid / Config::TN_THREADS_PER_ROW_A) + i * Config::TN_ROWS_PER_ITER_A; // K
                int c = (tid % Config::TN_THREADS_PER_ROW_A) * 4; // M
                int sc = c ^ ((r & 7) << 3); // Swizzle along M dimension for 128
                
                uint32_t sm_a = __cvta_generic_to_shared(&As[r * Config::BM + sc]);
                int gk = ko + r, gm = by * Config::BM + c;
                if constexpr (IsAligned) {
                    int bytes = (gk < K && gm < M) ? max(0, min(16, (M - gm) * 4)) : 0;
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm_a), "l"((const void*)gA_ptr[i]), "r"(bytes));
                } else {
                    float4 val = {0,0,0,0};
                    if (gk < K) {
                        if (gm < M) val.x = gA_ptr[i][0];
                        if (gm + 1 < M) val.y = gA_ptr[i][1];
                        if (gm + 2 < M) val.z = gA_ptr[i][2];
                        if (gm + 3 < M) val.w = gA_ptr[i][3];
                    }
                    *(float4*)&As[r * Config::BM + sc] = val;
                }
                gA_ptr[i] += Config::BK * lda;
            }
        }

        if constexpr (Layout == SgemmLayout::NT) {
            #pragma unroll
            for (int i = 0; i < Config::NT_LOAD_ITERS_B; i++) {
                int r = (tid / Config::NT_THREADS_PER_ROW_B) + i * Config::NT_ROWS_PER_ITER_B;
                int c = (tid % Config::NT_THREADS_PER_ROW_B) * 4;
                int sc = c ^ (((r & 7) << 2) & (Config::BK - 1));

                uint32_t sm_b = __cvta_generic_to_shared(&Bs[r * Config::BK + sc]);
                int gr = bx * Config::BN + r, gc = ko + c;
                if constexpr (IsAligned) {
                    int bytes = (gr < N && gc < K) ? max(0, min(16, (K - gc) * 4)) : 0;
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm_b), "l"((const void*)gB_ptr[i]), "r"(bytes));
                } else {
                    float4 val = {0,0,0,0};
                    if (gr < N && gc < K) {
                        val.x = gB_ptr[i][0]; if (gc+1 < K) val.y = gB_ptr[i][1];
                        if (gc+2 < K) val.z = gB_ptr[i][2]; if (gc+3 < K) val.w = gB_ptr[i][3];
                    }
                    *(float4*)&Bs[r * Config::BK + sc] = val;
                }
                gB_ptr[i] += Config::BK;
            }
        } else { // TN or NN
            #pragma unroll
            for (int i = 0; i < Config::TN_LOAD_ITERS_B; i++) {
                int r = (tid / Config::TN_THREADS_PER_ROW_B) + i * Config::TN_ROWS_PER_ITER_B; // K
                int c = (tid % Config::TN_THREADS_PER_ROW_B) * 4; // N
                int sc = c ^ (((r & 7) << 2) & (Config::BN - 1));
                
                uint32_t sm_b = __cvta_generic_to_shared(&Bs[r * Config::BN + sc]);
                int gk = ko + r, gn = bx * Config::BN + c;
                if constexpr (IsAligned) {
                    int bytes = (gk < K && gn < N) ? max(0, min(16, (N - gn) * 4)) : 0;
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm_b), "l"((const void*)gB_ptr[i]), "r"(bytes));
                } else {
                    float4 val = {0,0,0,0};
                    if (gk < K) {
                        if (gn < N) val.x = gB_ptr[i][0];
                        if (gn + 1 < N) val.y = gB_ptr[i][1];
                        if (gn + 2 < N) val.z = gB_ptr[i][2];
                        if (gn + 3 < N) val.w = gB_ptr[i][3];
                    }
                    *(float4*)&Bs[r * Config::BN + sc] = val;
                }
                gB_ptr[i] += Config::BK * ldb;
            }
        }
    };

    // -------------------------------------------------------------
    // WMMA Loads
    // -------------------------------------------------------------
    const int g_sh = lane / 4, t_sh = lane % 4;
    
    // Y-axis wmma pointers
    int rbaseA[Config::MMA_M], maskA[Config::MMA_M];
    #pragma unroll
    for (int i = 0; i < Config::MMA_M; i++) {
        rbaseA[i] = (wy * Config::WARP_TILE_M + i * 16 + g_sh) * ((Layout == SgemmLayout::NT || Layout == SgemmLayout::NN) ? Config::BK : 1);
        if constexpr (Layout == SgemmLayout::NT || Layout == SgemmLayout::NN) {
            maskA[i] = (((wy * Config::WARP_TILE_M + i * 16 + g_sh) & 7) << 2) & (Config::BK - 1);
        } else { // TN
            maskA[i] = (((wy * Config::WARP_TILE_M + i * 16 + g_sh) & 7) << 3) & (Config::BM - 1);
        }
    }
    
    // X-axis wmma pointers
    int rbaseB[Config::MMA_N];
    #pragma unroll
    for (int j = 0; j < Config::MMA_N; j++) rbaseB[j] = (wx * Config::WARP_TILE_N + j * 8 + g_sh) * ((Layout == SgemmLayout::NT) ? Config::BK : 1);

    auto load_frA = [&](uint32_t reg[4], int ks, int mi, int st) {
        float* As = smem + st * Config::STAGE_SIZE;
        if constexpr (Layout == SgemmLayout::NT || Layout == SgemmLayout::NN) {
            auto ga = [&](int row_idx, int c) { return *(uint32_t*)&As[row_idx + (c ^ maskA[mi])]; };
            reg[0] = ga(rbaseA[mi], ks + t_sh);
            reg[1] = ga(rbaseA[mi] + 8 * Config::BK, ks + t_sh);
            reg[2] = ga(rbaseA[mi], ks + t_sh + 4);
            reg[3] = ga(rbaseA[mi] + 8 * Config::BK, ks + t_sh + 4);
        } else {
            const int k0 = ks + t_sh, k4 = k0 + 4, row = (wy * Config::WARP_TILE_M + mi * 16 + g_sh);
            auto ga = [&](int k, int m) { return *(uint32_t*)&As[k * Config::BM + (m ^ ((k & 7) << 3))]; };
            reg[0] = ga(k0, row);
            reg[1] = ga(k0, row + 8);
            reg[2] = ga(k4, row);
            reg[3] = ga(k4, row + 8);
        }
    };

    auto load_frB = [&](uint32_t reg[2], int ks, int ni, int st) {
        float* Bs = smem + st * Config::STAGE_SIZE + Config::AS_SIZE;
        if constexpr (Layout == SgemmLayout::NT) {
            const int row = rbaseB[ni], mask = (((row / Config::BK) & 7) << 2) & (Config::BK - 1);
            auto gb = [&](int r, int c) { return *(uint32_t*)&Bs[r + (c ^ mask)]; };
            reg[0] = gb(row, ks + t_sh); reg[1] = gb(row, ks + t_sh + 4);
        } else {
            const int k0 = ks + t_sh, k4 = k0 + 4, col = (wx * Config::WARP_TILE_N + ni * 8 + g_sh);
            auto gb = [&](int k, int n) { return *(uint32_t*)&Bs[k * Config::BN + (n ^ ((k & 7) << 2))]; };
            reg[0] = gb(k0, col); reg[1] = gb(k4, col);
        }
    };

    if (k_start < k_end) {
        load_to_stage(0, k_start); asm volatile("cp.async.commit_group;\n");
        #pragma unroll
        for (int s = 1; s < Config::STAGES - 1; s++) {
            if (k_start + s * Config::BK < k_end) load_to_stage(s, k_start + s * Config::BK); 
            asm volatile("cp.async.commit_group;\n");
        }
    }

    int ws = Config::STAGES - 1, rs = 0; 
    uint32_t frA[2][Config::MMA_M][4], frB[2][Config::MMA_N][2];
    
    asm volatile("cp.async.wait_group %0;\n" :: "n"(Config::STAGES - 2)); __syncthreads();
    #pragma unroll
    for (int i = 0; i < Config::MMA_M; i++) load_frA(frA[0][i], 0, i, rs);
    #pragma unroll
    for (int j = 0; j < Config::MMA_N; j++) load_frB(frB[0][j], 0, j, rs);

    for (int k = k_start; k < k_end; k += Config::BK) {
        if (k + (Config::STAGES - 1) * Config::BK < k_end) load_to_stage(ws, k + (Config::STAGES - 1) * Config::BK); 
        asm volatile("cp.async.commit_group;\n");
        #pragma unroll
        for (int ks = 0; ks < Config::BK; ks += 8) {
            const int p = (ks / 8) & 1, q = p ^ 1;
            #pragma unroll
            for (int i = 0; i < Config::MMA_M; i++) {
                if (ks + 8 < Config::BK) { 
                    load_frA(frA[q][i], ks + 8, i, rs); 
                    for (int j = i; j < Config::MMA_N; j += Config::MMA_M) load_frB(frB[q][j], ks + 8, j, rs);
                }
                else if (k + Config::BK < k_end) { 
                    if (i == 0) { 
                        asm volatile("cp.async.wait_group %0;\n" :: "n"(Config::STAGES - 2)); 
                        __syncthreads(); 
                        rs = (rs+1)%Config::STAGES; ws = (ws+1)%Config::STAGES; 
                    }
                    load_frA(frA[q][i], 0, i, rs); 
                    for (int j = i; j < Config::MMA_N; j += Config::MMA_M) load_frB(frB[q][j], 0, j, rs);
                }
                #pragma unroll
                for (int j = 0; j < Config::MMA_N; j++) {
                    // Safety check if MMA_M < MMA_N/2 (e.g. 64x128 where MMA_M=4, MMA_N=16)
                    // We must load all B regs even if i exhausted.
                    if (ks + 8 < Config::BK && i == Config::MMA_M - 1 && Config::MMA_N/2 > Config::MMA_M) {
                        #pragma unroll
                        for(int rB = Config::MMA_M*2; rB < Config::MMA_N; rB++) load_frB(frB[q][rB], ks+8, rB, rs);
                    }
                    else if (k + Config::BK < k_end && i == Config::MMA_M - 1 && Config::MMA_N/2 > Config::MMA_M) {
                        #pragma unroll
                        for(int rB = Config::MMA_M*2; rB < Config::MMA_N; rB++) load_frB(frB[q][rB], 0, rB, rs);
                    }
                    
                    MMA_TF32(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3], 
                             frA[p][i][0], frA[p][i][1], frA[p][i][2], frA[p][i][3], 
                             frB[p][j][0], frB[p][j][1]);
                }
            }
        }
    }

    // Epilogue: float4 stores where possible, else float2
    // ----------------------------------------------------------------
    // Epilogue: C[r,c] = alpha * acc + beta * dst
    // ----------------------------------------------------------------
    const int g_epi = lane / 4, t_epi = lane % 4;
    float* dC_batch = C + (long long)batch * strideC;
    const bool can_vectorize_c = IsAligned && ((ldc & 1) == 0) && (((size_t)dC_batch & 7) == 0);

    #pragma unroll
    for (int i = 0; i < Config::MMA_M; i++) {
        const int r0 = by * Config::BM + wy * Config::WARP_TILE_M + i * 16 + g_epi;
        const int r8 = r0 + 8;
        if (r0 >= M) continue;

        #pragma unroll
        for (int j = 0; j < Config::MMA_N; j++) {
            const int c0 = bx * Config::BN + wx * Config::WARP_TILE_N + j * 8 + t_epi * 2;
            if (c0 >= N) continue;

            auto store_row = [&](int r, int c, float f0, float f1) {
                if (r >= M) return;
                float* dst = &dC_batch[(long long)r * ldc + c];
                if constexpr (SplitK > 1) {
                    atomicAdd(dst, alpha * f0);
                    if (c + 1 < N) atomicAdd(dst + 1, alpha * f1);
                } else {
                    if (can_vectorize_c && c + 1 < N) {
                        float2 old = (beta == 0.f) ? make_float2(0,0) : *(float2*)dst;
                        *(float2*)dst = { alpha * f0 + beta * old.x, alpha * f1 + beta * old.y };
                    } else {
                        dst[0] = alpha * f0 + (beta == 0 ? 0 : beta * dst[0]);
                        if (c + 1 < N) dst[1] = alpha * f1 + (beta == 0 ? 0 : beta * dst[1]);
                    }
                }
            };

            store_row(r0, c0, acc[i][j][0], acc[i][j][1]);
            store_row(r8, c0, acc[i][j][2], acc[i][j][3]);
        }
    }

}

template <typename Config>
__global__ void sgemm_scale_template_kernel(float* C, float beta, int M, int N, int ldc, long long strideC, int batchCount) {
    int r = blockIdx.y * blockDim.y + threadIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x, b = blockIdx.z;
    if (r < M && c < N && b < batchCount) {
        float* dst = &C[b * strideC + (long long)r * ldc + c];
        *dst = (beta == 0.f) ? 0.f : (*dst * beta);
    }
}

