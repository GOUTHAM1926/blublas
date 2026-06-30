#include "mycublas.h"
#include <cuda_runtime.h>
#include <stdint.h>

// ============================================================
// Strmm — Single-Precision Triangular Matrix Multiply
//
// Computes one of:
//   C = alpha * op(A) * B    (Left)
//   C = alpha * B * op(A)    (Right)
//
// A : triangular (M×M Left, N×N Right)
// B : dense M×N input
// C : dense M×N output
// op(A) = A (NoTrans) or A^T (Trans)
//
// Architecture
// ─────────────
//   Tile : 128×128×16  (BM × BN × BK)
//   Threads : 128 (4 warps)
//   Pipeline : 3-stage cp.async
//   Compute : mma.sync.aligned.m16n8k8  TF32→FP32
//   SMEM : dynamic, bank-conflict-free XOR swizzle
//   K-range optimisation : skip tiles entirely outside triangle
//
// Template parameters
//   Side  : 0=Left, 1=Right
//   Fill  : 0=Lower, 1=Upper
//   Trans : 0=NoTrans, 1=Trans
//   Diag  : 0=NonUnit, 1=Unit
//   IsAligned : 16-byte-aligned dense matrix path
// ============================================================

#define TRMM_BM      128
#define TRMM_BN      128
#define TRMM_BK      16
#define TRMM_STAGES  3
#define TRMM_THREADS 128
#define TRMM_AS      (TRMM_BM * TRMM_BK)
#define TRMM_BS      (TRMM_BK * TRMM_BN)
#define TRMM_STAGE   (TRMM_AS + TRMM_BS)

#define MMA_TF32(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1)                  \
    asm volatile(                                                     \
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "       \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"         \
        : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)                       \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1))

// ── Triangular element mask ───────────────────────────────────────────────
// (row_opA, col_opA) are coordinates inside op(A).
// Trans=0 → op(A)[r,c] = A[r,c];  Trans=1 → op(A)[r,c] = A[c,r]
// Returns: 1.0f (unit diag), original val (in-triangle), 0.0f (out-of-triangle)
template<int Fill, int Trans, int Diag>
__device__ __forceinline__
float apply_tri_mask(float val, int row_opA, int col_opA)
{
    // Map op(A) indices back to original A storage indices
    const int rA = (Trans == 0) ? row_opA : col_opA;
    const int cA = (Trans == 0) ? col_opA : row_opA;

    if constexpr (Diag == 1) {
        if (rA == cA) return 1.0f;
    }

    bool in_tri = (Fill == 0) ? (cA <= rA) : (cA >= rA);
    return in_tri ? val : 0.0f;
}

// ── Main kernel ───────────────────────────────────────────────────────────
template<int Side, int Fill, int Trans, int Diag, bool IsAligned>
__global__ void __launch_bounds__(TRMM_THREADS, 2)
strmm_kernel(
    int M, int N, int K,           // B is M×N; K = M (Left) or N (Right)
    float alpha,
    const float* __restrict__ A, int lda,
    const float* __restrict__ B, int ldb,
    float* __restrict__ C, int ldc)
{
    // ── CTA index with swizzle ────────────────────────────────────────────
    int bx = blockIdx.x, by = blockIdx.y;
    const int swizzle = 8;
    if (gridDim.y % swizzle == 0) {
        const int bi = blockIdx.y * gridDim.x + blockIdx.x;
        by = (bi % swizzle) + (bi / (gridDim.x * swizzle)) * swizzle;
        bx = (bi / swizzle) % gridDim.x;
    }
    if (by * TRMM_BM >= M || bx * TRMM_BN >= N) return;

    const int tid  = (int)threadIdx.x;
    const int lane = tid & 31, wid = tid >> 5;
    const int wy = wid >> 1, wx = wid & 1;

    // ── Effective K range (skip tiles entirely outside triangle) ──────────
    // For Left:  A is M×M; K==M
    // For Right: A is N×N; K==N
    int k_lo = 0, k_hi = K;

    if constexpr (Side == 0) {
        // Left Lower NoTrans:   C[i,j] = Σ_{k≤i} A[i,k]*B[k,j]
        //   → max useful k is (by+1)*BM
        // Left Upper Trans:     op(A)[i,k]=A[k,i], Upper A → k≤i
        //   → same k_hi bound
        if constexpr ((Fill == 0 && Trans == 0) || (Fill == 1 && Trans == 1))
            k_hi = min(K, (by + 1) * TRMM_BM);
        else  // Left Upper NoTrans or Left Lower Trans: start at by*BM
            k_lo = (by * TRMM_BM / TRMM_BK) * TRMM_BK;
    } else {
        // Right mode: sense is OPPOSITE to Left (bx drives the K bound, not by)
        // Right Lower Trans  (Fill=0,T=1): A^T[k,j]=A[j,k], Lower A→k≤j → k_hi
        // Right Upper NoTrans(Fill=1,T=0): A[k,j]=0 if k>j              → k_hi
        // Right Lower NoTrans(Fill=0,T=0): A[k,j]=0 if k<j              → k_lo
        // Right Upper Trans  (Fill=1,T=1): A^T[k,j]=A[j,k], Upper A→k≥j → k_lo
        if constexpr ((Fill == 0 && Trans == 1) || (Fill == 1 && Trans == 0))
            k_hi = min(K, (bx + 1) * TRMM_BN);
        else  // Right Lower NoTrans or Right Upper Trans: k_lo = bx*BN
            k_lo = (bx * TRMM_BN / TRMM_BK) * TRMM_BK;
    }

    // ── Accumulators ─────────────────────────────────────────────────────
    float acc[4][8][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 8; j++)
            for (int d = 0; d < 4; d++) acc[i][j][d] = 0.f;

    // ── Shared memory ─────────────────────────────────────────────────────
    extern __shared__ float smem[];

    // ── Global pointer induction for the DENSE matrix ────────────────────
    // In Left  mode: dense = B  →  As slots = A (triangular), Bs slots = B (dense)
    //   B tile rows: k_lo + r_thread; cols: bx*BN + c_thread
    // In Right mode: dense = B  →  As slots = B (dense), Bs slots = A (triangular)
    //   B tile rows: by*BM + r_thread; cols: k_lo + c_thread (for As)
    //
    // We induct pointers only for the dense side; triangular side uses
    // direct index computation (needed anyway for mask evaluation).
    //
    // gD_ptr[4]: dense-matrix pointers used by cp.async path (As or Bs)
    const float* gD_ptr[4];

    if constexpr (Side == 0) {
        // Dense = B, loaded into Bs[BK][BN]
        // Bs row r = k tile row: base = k_lo + r_tile, r_tile = i*4 + tid/32
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int r = i * 4 + (tid / 32);
            gD_ptr[i] = B + (long long)(k_lo + r) * ldb + (bx * TRMM_BN + (tid % 32) * 4);
        }
    } else {
        // Dense = B, loaded into As[BM][BK]
        // As row r = output row: by*BM + r_tile, r_tile = tid/4 + i*32
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int r = (tid / 4) + i * 32;
            gD_ptr[i] = B + (long long)(by * TRMM_BM + r) * ldb + (k_lo + (tid % 4) * 4);
        }
    }

    // ── load_to_stage lambda ──────────────────────────────────────────────
    // Loads one BM×BK (As) + BK×BN (Bs) tile from global to smem[stage].
    // As holds the LEFT multiply operand, Bs the RIGHT.
    //   Left  mode: As ← op(A)[by*BM:+BM, k:+BK]  (triangular, masked, scalar)
    //               Bs ← B[k:+BK, bx*BN:+BN]       (dense, cp.async)
    //   Right mode: As ← B[by*BM:+BM, k:+BK]       (dense, cp.async)
    //               Bs ← op(A)[k:+BK, bx*BN:+BN]   (triangular, masked, scalar)
    auto load_to_stage = [&](int stage, int k_curr) __attribute__((always_inline)) {
        float* As = smem + stage * TRMM_STAGE;
        float* Bs = As + TRMM_AS;

        // ── As tile ───────────────────────────────────────────────────────
        if constexpr (Side == 0) {
            // Left: As ← op(A)[by*BM+r, k_curr+c]  — scalar masked load
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int r  = (tid / 4) + i * 32;
                const int c  = (tid % 4) * 4;
                const int sc = c ^ ((r & 3) << 2);
                const int gr = by * TRMM_BM + r;
                float4 val = {0.f, 0.f, 0.f, 0.f};
                if (gr < M) {
                    #pragma unroll
                    for (int d = 0; d < 4; d++) {
                        const int gc = k_curr + c + d;
                        if (gc < K) {
                            float raw;
                            if constexpr (Trans == 0)
                                raw = A[(long long)gr * lda + gc];
                            else
                                raw = A[(long long)gc * lda + gr];
                            (&val.x)[d] = apply_tri_mask<Fill, Trans, Diag>(raw, gr, gc);
                        }
                    }
                }
                *(float4*)&As[r * TRMM_BK + sc] = val;
            }
        } else {
            // Right: As ← B[by*BM+r, k_curr+c]  — dense cp.async
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int r  = (tid / 4) + i * 32;
                const int c  = (tid % 4) * 4;
                const int sc = c ^ ((r & 3) << 2);
                uint32_t sm_a = __cvta_generic_to_shared(&As[r * TRMM_BK + sc]);
                const int gr  = by * TRMM_BM + r;
                const int gc  = k_curr + c;
                if constexpr (IsAligned) {
                    int bytes = (gr < M) ? max(0, min(16, (K - gc) * 4)) : 0;
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::
                        "r"(sm_a), "l"(gD_ptr[i]), "r"(bytes));
                } else {
                    float4 val = {0.f, 0.f, 0.f, 0.f};
                    if (gr < M) {
                        if (gc   < K) val.x = gD_ptr[i][0];
                        if (gc+1 < K) val.y = gD_ptr[i][1];
                        if (gc+2 < K) val.z = gD_ptr[i][2];
                        if (gc+3 < K) val.w = gD_ptr[i][3];
                    }
                    *(float4*)&As[r * TRMM_BK + sc] = val;
                }
                gD_ptr[i] += TRMM_BK;  // pointer induction
            }
        }

        // ── Bs tile ───────────────────────────────────────────────────────
        if constexpr (Side == 1) {
            // Right: Bs ← op(A)[k_curr+r, bx*BN+c]  — scalar masked load
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int r      = i * 4 + (tid / 32);
                const int c_base = (tid % 32) * 4;
                const int sc     = c_base ^ ((r & 7) << 2);
                const int gr     = k_curr + r;  // row of op(A)
                if (r < TRMM_BK) {
                    float4 val = {0.f, 0.f, 0.f, 0.f};
                    if (gr < K) {
                        #pragma unroll
                        for (int d = 0; d < 4; d++) {
                            const int gc = bx * TRMM_BN + c_base + d;
                            if (gc < N) {
                                float raw;
                                if constexpr (Trans == 0)
                                    raw = A[(long long)gr * lda + gc];
                                else
                                    raw = A[(long long)gc * lda + gr];
                                (&val.x)[d] = apply_tri_mask<Fill, Trans, Diag>(raw, gr, gc);
                            }
                        }
                    }
                    *(float4*)&Bs[r * TRMM_BN + sc] = val;
                }
            }
        } else {
            // Left: Bs ← B[k_curr+r, bx*BN+c]  — dense cp.async
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int r      = i * 4 + (tid / 32);
                const int c_base = (tid % 32) * 4;
                const int sc     = c_base ^ ((r & 7) << 2);
                uint32_t sm_b    = __cvta_generic_to_shared(&Bs[r * TRMM_BN + sc]);
                const int gr     = k_curr + r;
                const int gc     = bx * TRMM_BN + c_base;
                if (r < TRMM_BK) {
                    if constexpr (IsAligned) {
                        int bytes = (gr < K) ? max(0, min(16, (N - gc) * 4)) : 0;
                        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::
                            "r"(sm_b), "l"(gD_ptr[i]), "r"(bytes));
                    } else {
                        float4 val = {0.f, 0.f, 0.f, 0.f};
                        if (gr < K) {
                            if (gc   < N) val.x = gD_ptr[i][0];
                            if (gc+1 < N) val.y = gD_ptr[i][1];
                            if (gc+2 < N) val.z = gD_ptr[i][2];
                            if (gc+3 < N) val.w = gD_ptr[i][3];
                        }
                        *(float4*)&Bs[r * TRMM_BN + sc] = val;
                    }
                    gD_ptr[i] += (long long)TRMM_BK * ldb;  // pointer induction
                }
            }
        }
    };

    // ── SMEM → register load helpers (identical to v18) ──────────────────
    const int g_sh = lane / 4, t_sh = lane % 4;

    int rbaseA[4], maskA[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        rbaseA[i] = (wy * 64 + i * 16 + g_sh) * TRMM_BK;
        maskA[i]  = ((wy * 64 + i * 16 + g_sh) & 3) << 2;
    }
    int cbaseB[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) cbaseB[j] = wx * 64 + j * 8 + g_sh;

    auto load_frA = [&](uint32_t reg[4], int ks, int mi, int st) {
        float* As = smem + st * TRMM_STAGE;
        auto ga = [&](int row_off, int c) {
            return *(uint32_t*)&As[row_off + (c ^ maskA[mi])];
        };
        reg[0] = ga(rbaseA[mi],             ks + t_sh);
        reg[1] = ga(rbaseA[mi] + 8*TRMM_BK, ks + t_sh);
        reg[2] = ga(rbaseA[mi],             ks + t_sh + 4);
        reg[3] = ga(rbaseA[mi] + 8*TRMM_BK, ks + t_sh + 4);
    };

    auto load_frB = [&](uint32_t reg[2], int ks, int ni, int st) {
        float* Bs = smem + st * TRMM_STAGE + TRMM_AS;
        const int r0 = ks + t_sh, r1 = r0 + 4;
        auto gb = [&](int r, int c) {
            return *(uint32_t*)&Bs[r * TRMM_BN + (c ^ ((r & 7) << 2))];
        };
        reg[0] = gb(r0, cbaseB[ni]);
        reg[1] = gb(r1, cbaseB[ni]);
    };

    // ── 3-Stage pipeline ──────────────────────────────────────────────────
    load_to_stage(0, k_lo);
    asm volatile("cp.async.commit_group;\n");
    if (k_lo + TRMM_BK < k_hi) load_to_stage(1, k_lo + TRMM_BK);
    asm volatile("cp.async.commit_group;\n");

    int write_stage = 2, read_stage = 0;
    uint32_t frA[2][4][4], frB[2][8][2];

    asm volatile("cp.async.wait_group 1;\n");
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 4; i++) load_frA(frA[0][i], 0, i, read_stage);
    #pragma unroll
    for (int j = 0; j < 8; j++) load_frB(frB[0][j], 0, j, read_stage);

    // ── Main K loop ───────────────────────────────────────────────────────
    for (int k = k_lo; k < k_hi; k += TRMM_BK) {

        // Issue next async load
        if (k + 2 * TRMM_BK < k_hi)
            load_to_stage(write_stage, k + 2 * TRMM_BK);
        asm volatile("cp.async.commit_group;\n");

        // Compute BK steps, double-buffering register files
        #pragma unroll
        for (int ks = 0; ks < TRMM_BK; ks += 8) {
            const int p = (ks / 8) & 1, q = p ^ 1;

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                if (ks + 8 < TRMM_BK) {
                    // Prefetch next ks tile from same smem stage
                    load_frA(frA[q][i], ks + 8, i, read_stage);
                    load_frB(frB[q][i * 2    ], ks + 8, i * 2,     read_stage);
                    load_frB(frB[q][i * 2 + 1], ks + 8, i * 2 + 1, read_stage);
                } else if (k + TRMM_BK < k_hi) {
                    // Stage boundary: wait for next stage, advance ring
                    if (i == 0) {
                        asm volatile("cp.async.wait_group 1;\n");
                        __syncthreads();
                        read_stage  = (read_stage  + 1) % TRMM_STAGES;
                        write_stage = (write_stage + 1) % TRMM_STAGES;
                    }
                    load_frA(frA[q][i], 0, i, read_stage);
                    load_frB(frB[q][i * 2    ], 0, i * 2,     read_stage);
                    load_frB(frB[q][i * 2 + 1], 0, i * 2 + 1, read_stage);
                }

                // MMA: accumulate 4 rows × 8 cols × 4 elements
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    MMA_TF32(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3],
                             frA[p][i][0], frA[p][i][1], frA[p][i][2], frA[p][i][3],
                             frB[p][j][0], frB[p][j][1]);
                }
            }
        }
    }

    // ── Epilogue: write C = alpha * acc ──────────────────────────────────
    const int g_epi = lane / 4, t_epi = lane % 4;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int r0  = by * TRMM_BM + wy * 64 + i * 16 + g_epi;
            const int r8  = r0 + 8;
            const int col = bx * TRMM_BN + wx * 64 + j * 8 + t_epi * 2;

            auto store2 = [&](int r, int c, float f0, float f1) {
                if (r >= M || c >= N) return;
                float* dst = &C[(long long)r * ldc + c];
                if (c + 1 < N && (((size_t)dst & 7) == 0))
                    *(float2*)dst = make_float2(alpha * f0, alpha * f1);
                else {
                    dst[0] = alpha * f0;
                    if (c + 1 < N) dst[1] = alpha * f1;
                }
            };

            store2(r0, col, acc[i][j][0], acc[i][j][1]);
            store2(r8, col, acc[i][j][2], acc[i][j][3]);
        }
    }
}

// ── Host dispatcher ───────────────────────────────────────────────────────

// Instantiate all 16 Side×Fill×Trans×Diag combinations for the dispatch table.
// IsAligned is chosen at runtime based on pointer/stride alignment.

#define INST(S, F, T, D)                                                    \
    template __global__ void strmm_kernel<S,F,T,D,true>(                   \
        int,int,int,float,const float*,int,const float*,int,float*,int);    \
    template __global__ void strmm_kernel<S,F,T,D,false>(                  \
        int,int,int,float,const float*,int,const float*,int,float*,int);

INST(0,0,0,0) INST(0,0,0,1) INST(0,0,1,0) INST(0,0,1,1)
INST(0,1,0,0) INST(0,1,0,1) INST(0,1,1,0) INST(0,1,1,1)
INST(1,0,0,0) INST(1,0,0,1) INST(1,0,1,0) INST(1,0,1,1)
INST(1,1,0,0) INST(1,1,0,1) INST(1,1,1,0) INST(1,1,1,1)

// Generic launch helper
template<int Side, int Fill, int Trans, int Diag>
static void launch_strmm(
    cudaStream_t stream,
    int M, int N, int K, float alpha,
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc)
{
    const bool aligned =
        (((size_t)A & 15) == 0) && ((lda & 3) == 0) &&
        (((size_t)B & 15) == 0) && ((ldb & 3) == 0);

    static const size_t smem = TRMM_STAGES * TRMM_STAGE * sizeof(float);

    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(strmm_kernel<Side,Fill,Trans,Diag,true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaFuncSetAttribute(strmm_kernel<Side,Fill,Trans,Diag,false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        attr_set = true;
    }

    dim3 grid((N + TRMM_BN - 1) / TRMM_BN, (M + TRMM_BM - 1) / TRMM_BM, 1);

    if (aligned)
        strmm_kernel<Side,Fill,Trans,Diag,true>
            <<<grid, TRMM_THREADS, smem, stream>>>(M,N,K,alpha,A,lda,B,ldb,C,ldc);
    else
        strmm_kernel<Side,Fill,Trans,Diag,false>
            <<<grid, TRMM_THREADS, smem, stream>>>(M,N,K,alpha,A,lda,B,ldb,C,ldc);
}

// ── Public C API ──────────────────────────────────────────────────────────
// Matches BLAS STRMM signature:
//   side  : 'L'/'l' = Left,    'R'/'r' = Right
//   uplo  : 'U'/'u' = Upper,   'L'/'l' = Lower
//   transa: 'N'/'n' = NoTrans, 'T'/'t' or 'C'/'c' = Trans
//   diag  : 'N'/'n' = NonUnit, 'U'/'u' = Unit
//   M, N  : dimensions of B (and C)
//   alpha : scalar
//   A     : triangular matrix (M×M if Left, N×N if Right)
//   lda   : leading dimension of A
//   B     : dense input  (M×N)
//   ldb   : leading dimension of B
//   C     : dense output (M×N)  [may alias B for in-place]
//   ldc   : leading dimension of C
extern "C" void mycublasStrmm(
    mycublasHandle_t handle,
    char side, char uplo, char transa, char diag,
    int M, int N, float alpha,
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc)
{
    if (M <= 0 || N <= 0) return;

    cudaStream_t stream = (handle != nullptr) ? handle->stream : 0;

    const int s = (side   == 'L' || side   == 'l') ? 0 : 1;
    const int f = (uplo   == 'L' || uplo   == 'l') ? 0 : 1;
    const int t = (transa == 'N' || transa == 'n') ? 0 : 1;
    const int d = (diag   == 'U' || diag   == 'u') ? 1 : 0;
    const int K = (s == 0) ? M : N;

    // Runtime dispatch over 16 specialisations
    switch ((s << 3) | (f << 2) | (t << 1) | d) {
        case 0b0000: launch_strmm<0,0,0,0>(stream,M,N,K,alpha,A,lda,B,ldb,C,ldc); break;
        case 0b0001: launch_strmm<0,0,0,1>(stream,M,N,K,alpha,A,lda,B,ldb,C,ldc); break;
        case 0b0010: launch_strmm<0,0,1,0>(stream,M,N,K,alpha,A,lda,B,ldb,C,ldc); break;
        case 0b0011: launch_strmm<0,0,1,1>(stream,M,N,K,alpha,A,lda,B,ldb,C,ldc); break;
        case 0b0100: launch_strmm<0,1,0,0>(stream,M,N,K,alpha,A,lda,B,ldb,C,ldc); break;
        case 0b0101: launch_strmm<0,1,0,1>(stream,M,N,K,alpha,A,lda,B,ldb,C,ldc); break;
        case 0b0110: launch_strmm<0,1,1,0>(stream,M,N,K,alpha,A,lda,B,ldb,C,ldc); break;
        case 0b0111: launch_strmm<0,1,1,1>(stream,M,N,K,alpha,A,lda,B,ldb,C,ldc); break;
        case 0b1000: launch_strmm<1,0,0,0>(stream,M,N,K,alpha,A,lda,B,ldb,C,ldc); break;
        case 0b1001: launch_strmm<1,0,0,1>(stream,M,N,K,alpha,A,lda,B,ldb,C,ldc); break;
        case 0b1010: launch_strmm<1,0,1,0>(stream,M,N,K,alpha,A,lda,B,ldb,C,ldc); break;
        case 0b1011: launch_strmm<1,0,1,1>(stream,M,N,K,alpha,A,lda,B,ldb,C,ldc); break;
        case 0b1100: launch_strmm<1,1,0,0>(stream,M,N,K,alpha,A,lda,B,ldb,C,ldc); break;
        case 0b1101: launch_strmm<1,1,0,1>(stream,M,N,K,alpha,A,lda,B,ldb,C,ldc); break;
        case 0b1110: launch_strmm<1,1,1,0>(stream,M,N,K,alpha,A,lda,B,ldb,C,ldc); break;
        case 0b1111: launch_strmm<1,1,1,1>(stream,M,N,K,alpha,A,lda,B,ldb,C,ldc); break;
    }
}
