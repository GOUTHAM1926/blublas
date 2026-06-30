#include "mycublas.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include <unordered_map>

// ============================================================
// 64x64 Tile GEMM V18  (NN / NT / TN)
//
// Tile: BM=64, BN=64, BK=16, THREADS=128, STAGES=3
// Smem per block: 3 * (64*16 + 16*64) * 4 = 3 * 2048 * 4 = 24 KB
// vs 128x128: 48 KB -> fits 4 blocks/SM instead of 2 -> 2x occupancy
//
// Warp layout: 2x2 (wy=wid>>1, wx=wid&1)
//   Each warp: 32x32 output  (2 M-tiles of m16n8k8 x 4 N-tiles)
//
// Use when: gx64*gy64 < 80  (too few 128x128 blocks for full SM coverage)
// SplitK variant for K>=512 when gx64*gy64 < 80
// ============================================================

#define T64_BM      64
#define T64_BN      64
#define T64_BK      16
#define T64_STAGES  3
#define T64_THREADS 128
#define T64_AS      (T64_BM * T64_BK)   // 64*16 = 1024
#define T64_BS      (T64_BK * T64_BN)   // 16*64 = 1024
#define T64_SS      (T64_AS + T64_BS)   // 2048 per stage
// smem = 3 * 2048 * 4 = 24576 bytes

#ifndef MMA_TF32
#define MMA_TF32(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1)                \
    asm volatile(                                                   \
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "     \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"       \
        : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)                     \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1))
#endif

// ============================================================
// NN  64x64 kernel  (C = A*B,  A[M,K] row-major, B[K,N] row-major)
// ============================================================
template <bool IsAligned, int SplitK>
__global__ void __launch_bounds__(T64_THREADS, 2)
sgemm_nn_tile64_kernel(
    int M, int N, int K, float alpha,
    const float* __restrict__ A, int lda, long long strideA,
    const float* __restrict__ B, int ldb, long long strideB,
    float beta,
    float* __restrict__ C, int ldc, long long strideC,
    int batchCount)
{
    const int batch  = blockIdx.z / SplitK;
    const int sk_idx = blockIdx.z % SplitK;
    if (batch >= batchCount) return;

    int bx = blockIdx.x, by = blockIdx.y;
    const int swizzle = 4;
    if (gridDim.y % swizzle == 0) {
        const int bi = blockIdx.y * gridDim.x + blockIdx.x;
        by = (bi % swizzle) + (bi / (gridDim.x * swizzle)) * swizzle;
        bx = (bi / swizzle) % gridDim.x;
    }
    if (by * T64_BM >= M || bx * T64_BN >= N) return;

    const int tid  = threadIdx.x;
    const int lane = tid & 31, wid = tid >> 5;
    const int wy   = wid >> 1, wx = wid & 1;

    const int k_tiles      = (K + T64_BK - 1) / T64_BK;
    const int tiles_per_sk = (k_tiles + SplitK - 1) / SplitK;
    const int k_start      = sk_idx * tiles_per_sk * T64_BK;
    const int k_end        = min(K, k_start + tiles_per_sk * T64_BK);

    const float* gA_ptr[2];
    const float* gB_ptr[2];
    for (int i = 0; i < 2; i++) {
        int rA = by * T64_BM + (tid / 4) + i * 32;
        gA_ptr[i] = A + (long long)batch * strideA + (long long)rA * lda + (k_start + (tid % 4) * 4);
        int rB = i * 8 + (tid / 16);
        gB_ptr[i] = B + (long long)batch * strideB + (long long)(k_start + rB) * ldb + (bx * T64_BN + (tid % 16) * 4);
    }

    extern __shared__ float smem[];
    float acc[2][4][4] = {};

    auto load_stage = [&](int stage, int ko) {
        float* As = smem + stage * T64_SS;
        float* Bs = As + T64_AS;
        for (int i = 0; i < 2; i++) {
            const int r = tid / 4 + i * 32, c = (tid % 4) * 4;
            const int sc = c ^ ((r & 3) << 2);
            uint32_t sm_a = __cvta_generic_to_shared(&As[r * T64_BK + sc]);
            const int gr = by * T64_BM + r, gc = ko + c;
            if constexpr (IsAligned) {
                int bytes = (gr < M) ? max(0, min(16, (K - gc) * 4)) : 0;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm_a), "l"(gA_ptr[i]), "r"(bytes));
            } else {
                float4 v = {0,0,0,0}; if (gr < M) { if (gc<K) v.x=gA_ptr[i][0]; if (gc+1<K) v.y=gA_ptr[i][1]; if (gc+2<K) v.z=gA_ptr[i][2]; if (gc+3<K) v.w=gA_ptr[i][3]; }
                *(float4*)&As[r * T64_BK + sc] = v;
            }
            gA_ptr[i] += T64_BK;
        }
        for (int i = 0; i < 2; i++) {
            const int r = i * 8 + tid / 16, c_base = (tid % 16) * 4;
            const int sc = c_base ^ ((r & 7) << 2);
            uint32_t sm_b = __cvta_generic_to_shared(&Bs[r * T64_BN + sc]);
            const int gr = ko + r, gc = bx * T64_BN + c_base;
            if constexpr (IsAligned) {
                int bytes = (gr < K) ? max(0, min(16, (N - gc) * 4)) : 0;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm_b), "l"(gB_ptr[i]), "r"(bytes));
            } else {
                float4 v = {0,0,0,0}; if (gr < K) { if (gc<N) v.x=gB_ptr[i][0]; if (gc+1<N) v.y=gB_ptr[i][1]; if (gc+2<N) v.z=gB_ptr[i][2]; if (gc+3<N) v.w=gB_ptr[i][3]; }
                *(float4*)&Bs[r * T64_BN + sc] = v;
            }
            gB_ptr[i] += (long long)T64_BK * ldb;
        }
    };

    const int g_sh = lane / 4, t_sh = lane % 4;
    int rbaseA[2], maskA[2];
    for (int i = 0; i < 2; i++) { rbaseA[i] = (wy * 32 + i * 16 + g_sh) * T64_BK; maskA[i] = ((wy * 32 + i * 16 + g_sh) & 3) << 2; }
    int cbaseB[4];
    for (int j = 0; j < 4; j++) cbaseB[j] = wx * 32 + j * 8 + g_sh;

    auto load_frA = [&](uint32_t reg[4], int ks, int mi, int st) {
        float* As = smem + st * T64_SS;
        auto ga = [&](int ri, int c) { return *(uint32_t*)&As[ri + (c ^ maskA[mi])]; };
        reg[0] = ga(rbaseA[mi],          ks + t_sh);
        reg[1] = ga(rbaseA[mi] + 8*T64_BK, ks + t_sh);
        reg[2] = ga(rbaseA[mi],          ks + t_sh + 4);
        reg[3] = ga(rbaseA[mi] + 8*T64_BK, ks + t_sh + 4);
    };
    auto load_frB = [&](uint32_t reg[2], int ks, int ni, int st) {
        float* Bs = smem + st * T64_SS + T64_AS;
        auto gb = [&](int r, int c) { return *(uint32_t*)&Bs[r * T64_BN + (c ^ ((r & 7) << 2))]; };
        reg[0] = gb(ks + t_sh,     cbaseB[ni]);
        reg[1] = gb(ks + t_sh + 4, cbaseB[ni]);
    };

    if (k_start < k_end) {
        load_stage(0, k_start); asm volatile("cp.async.commit_group;\n");
        if (k_start + T64_BK < k_end) { load_stage(1, k_start + T64_BK); asm volatile("cp.async.commit_group;\n"); }
    }

    int ws = 2, rs = 0;
    uint32_t frA[2][2][4], frB[2][4][2];
    asm volatile("cp.async.wait_group 1;\n"); __syncthreads();
    for (int i = 0; i < 2; i++) load_frA(frA[0][i], 0, i, rs);
    for (int j = 0; j < 4; j++) load_frB(frB[0][j], 0, j, rs);

    for (int k = k_start; k < k_end; k += T64_BK) {
        if (k + 2 * T64_BK < k_end) { load_stage(ws, k + 2 * T64_BK); asm volatile("cp.async.commit_group;\n"); }
        #pragma unroll
        for (int ks = 0; ks < T64_BK; ks += 8) {
            const int p = (ks / 8) & 1, q = p ^ 1;
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                if (ks + 8 < T64_BK) {
                    load_frA(frA[q][i], ks + 8, i, rs);
                    load_frB(frB[q][i*2],   ks + 8, i*2,   rs);
                    load_frB(frB[q][i*2+1], ks + 8, i*2+1, rs);
                } else if (k + T64_BK < k_end) {
                    if (i == 0) { asm volatile("cp.async.wait_group 1;\n"); __syncthreads(); rs = (rs+1)%T64_STAGES; ws = (ws+1)%T64_STAGES; }
                    load_frA(frA[q][i], 0, i, rs);
                    load_frB(frB[q][i*2],   0, i*2,   rs);
                    load_frB(frB[q][i*2+1], 0, i*2+1, rs);
                }
                #pragma unroll
                for (int j = 0; j < 4; j++)
                    MMA_TF32(acc[i][j][0],acc[i][j][1],acc[i][j][2],acc[i][j][3],
                             frA[p][i][0],frA[p][i][1],frA[p][i][2],frA[p][i][3],
                             frB[p][j][0],frB[p][j][1]);
            }
        }
    }

    const int g_epi = lane / 4, t_epi = lane % 4;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            const int r0  = by * T64_BM + wy * 32 + i * 16 + g_epi;
            const int r8  = r0 + 8;
            const int col = bx * T64_BN + wx * 32 + j * 8 + t_epi * 2;
            float* dC = C + (long long)batch * strideC;
            auto sf2 = [&](int r, int c, float f0, float f1) {
                if (r >= M || c >= N) return;
                float* dst = &dC[(long long)r * ldc + c];
                if constexpr (SplitK > 1) {
                    atomicAdd(dst,     alpha * f0);
                    if (c+1 < N) atomicAdd(dst+1, alpha * f1);
                } else {
                    if (c+1 < N && (((size_t)dst & 7) == 0)) {
                        float2 old = (beta == 0.f) ? make_float2(0,0) : *(float2*)dst;
                        *(float2*)dst = { alpha*f0 + beta*old.x, alpha*f1 + beta*old.y };
                    } else {
                        dst[0] = alpha*f0 + (beta==0.f ? 0.f : beta*dst[0]);
                        if (c+1<N) dst[1] = alpha*f1 + (beta==0.f ? 0.f : beta*dst[1]);
                    }
                }
            };
            sf2(r0, col, acc[i][j][0], acc[i][j][1]);
            sf2(r8, col, acc[i][j][2], acc[i][j][3]);
        }
    }
}

// ============================================================
// NT  64x64 kernel  (C = A*B^T,  A[M,K] row-major, B[N,K] row-major)
// ============================================================
template <bool IsAligned, int SplitK>
__global__ void __launch_bounds__(T64_THREADS, 2)
sgemm_nt_tile64_kernel(
    int M, int N, int K, float alpha,
    const float* __restrict__ A, int lda, long long strideA,
    const float* __restrict__ B, int ldb, long long strideB,
    float beta,
    float* __restrict__ C, int ldc, long long strideC,
    int batchCount)
{
    const int batch  = blockIdx.z / SplitK;
    const int sk_idx = blockIdx.z % SplitK;
    if (batch >= batchCount) return;

    int bx = blockIdx.x, by = blockIdx.y;
    const int swizzle = 4;
    if (gridDim.y % swizzle == 0) {
        const int bi = blockIdx.y * gridDim.x + blockIdx.x;
        by = (bi % swizzle) + (bi / (gridDim.x * swizzle)) * swizzle;
        bx = (bi / swizzle) % gridDim.x;
    }
    if (by * T64_BM >= M || bx * T64_BN >= N) return;

    const int tid  = threadIdx.x;
    const int lane = tid & 31, wid = tid >> 5;
    const int wy   = wid >> 1, wx = wid & 1;

    const int k_tiles      = (K + T64_BK - 1) / T64_BK;
    const int tiles_per_sk = (k_tiles + SplitK - 1) / SplitK;
    const int k_start      = sk_idx * tiles_per_sk * T64_BK;
    const int k_end        = min(K, k_start + tiles_per_sk * T64_BK);

    const float* gA_ptr[2];
    const float* gB_ptr[2];
    for (int i = 0; i < 2; i++) {
        int rA = by * T64_BM + (tid / 4) + i * 32;
        gA_ptr[i] = A + (long long)batch * strideA + (long long)rA * lda + (k_start + (tid % 4) * 4);
        int rB = bx * T64_BN + (tid / 4) + i * 32;
        gB_ptr[i] = B + (long long)batch * strideB + (long long)rB * ldb + (k_start + (tid % 4) * 4);
    }

    extern __shared__ float smem[];
    float acc[2][4][4] = {};

    auto load_stage = [&](int stage, int ko) {
        float* As = smem + stage * T64_SS;
        float* Bs = As + T64_AS;
        for (int i = 0; i < 2; i++) {
            const int r = tid / 4 + i * 32, c = (tid % 4) * 4, sc = c ^ ((r & 3) << 2);
            uint32_t sm_a = __cvta_generic_to_shared(&As[r * T64_BK + sc]);
            const int gr = by * T64_BM + r, gc = ko + c;
            if constexpr (IsAligned) {
                int bytes = (gr < M) ? max(0, min(16, (K - gc) * 4)) : 0;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm_a), "l"(gA_ptr[i]), "r"(bytes));
            } else {
                float4 v={0,0,0,0}; if (gr<M){if(gc<K)v.x=gA_ptr[i][0];if(gc+1<K)v.y=gA_ptr[i][1];if(gc+2<K)v.z=gA_ptr[i][2];if(gc+3<K)v.w=gA_ptr[i][3];}
                *(float4*)&As[r * T64_BK + sc] = v;
            }
            gA_ptr[i] += T64_BK;
        }
        for (int i = 0; i < 2; i++) {
            const int r = tid / 4 + i * 32, c = (tid % 4) * 4, sc = c ^ ((r & 3) << 2);
            uint32_t sm_b = __cvta_generic_to_shared(&Bs[r * T64_BK + sc]);
            const int gr = bx * T64_BN + r, gc = ko + c;
            if constexpr (IsAligned) {
                int bytes = (gr < N) ? max(0, min(16, (K - gc) * 4)) : 0;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm_b), "l"(gB_ptr[i]), "r"(bytes));
            } else {
                float4 v={0,0,0,0}; if (gr<N){if(gc<K)v.x=gB_ptr[i][0];if(gc+1<K)v.y=gB_ptr[i][1];if(gc+2<K)v.z=gB_ptr[i][2];if(gc+3<K)v.w=gB_ptr[i][3];}
                *(float4*)&Bs[r * T64_BK + sc] = v;
            }
            gB_ptr[i] += T64_BK;
        }
    };

    const int g_sh = lane / 4, t_sh = lane % 4;
    int rbaseA[2], maskA[2];
    for (int i=0;i<2;i++){rbaseA[i]=(wy*32+i*16+g_sh)*T64_BK; maskA[i]=((wy*32+i*16+g_sh)&3)<<2;}
    int rbaseB[4], maskB[4];
    for (int j=0;j<4;j++){rbaseB[j]=(wx*32+j*8+g_sh)*T64_BK; maskB[j]=((wx*32+j*8+g_sh)&3)<<2;}

    auto load_frA = [&](uint32_t reg[4], int ks, int mi, int st) {
        float* As = smem + st * T64_SS;
        auto ga = [&](int ri, int c){return *(uint32_t*)&As[ri+(c^maskA[mi])];};
        reg[0]=ga(rbaseA[mi],         ks+t_sh); reg[1]=ga(rbaseA[mi]+8*T64_BK, ks+t_sh);
        reg[2]=ga(rbaseA[mi],         ks+t_sh+4); reg[3]=ga(rbaseA[mi]+8*T64_BK, ks+t_sh+4);
    };
    auto load_frB = [&](uint32_t reg[2], int ks, int ni, int st) {
        float* Bs = smem + st * T64_SS + T64_AS;
        auto gb = [&](int ri, int c){return *(uint32_t*)&Bs[ri+(c^maskB[ni])];};
        reg[0]=gb(rbaseB[ni],         ks+t_sh); reg[1]=gb(rbaseB[ni], ks+t_sh+4);
    };

    if (k_start < k_end) {
        load_stage(0, k_start); asm volatile("cp.async.commit_group;\n");
        if (k_start + T64_BK < k_end) { load_stage(1, k_start + T64_BK); asm volatile("cp.async.commit_group;\n"); }
    }
    int ws=2, rs=0;
    uint32_t frA[2][2][4], frB[2][4][2];
    asm volatile("cp.async.wait_group 1;\n"); __syncthreads();
    for (int i=0;i<2;i++) load_frA(frA[0][i],0,i,rs);
    for (int j=0;j<4;j++) load_frB(frB[0][j],0,j,rs);

    for (int k = k_start; k < k_end; k += T64_BK) {
        if (k + 2*T64_BK < k_end) { load_stage(ws, k+2*T64_BK); asm volatile("cp.async.commit_group;\n"); }
        #pragma unroll
        for (int ks = 0; ks < T64_BK; ks += 8) {
            const int p=(ks/8)&1, q=p^1;
            #pragma unroll
            for (int i=0;i<2;i++) {
                if (ks+8<T64_BK) {
                    load_frA(frA[q][i],ks+8,i,rs);
                    load_frB(frB[q][i*2],ks+8,i*2,rs); load_frB(frB[q][i*2+1],ks+8,i*2+1,rs);
                } else if (k+T64_BK<k_end) {
                    if(i==0){asm volatile("cp.async.wait_group 1;\n");__syncthreads();rs=(rs+1)%T64_STAGES;ws=(ws+1)%T64_STAGES;}
                    load_frA(frA[q][i],0,i,rs);
                    load_frB(frB[q][i*2],0,i*2,rs); load_frB(frB[q][i*2+1],0,i*2+1,rs);
                }
                #pragma unroll
                for (int j=0;j<4;j++)
                    MMA_TF32(acc[i][j][0],acc[i][j][1],acc[i][j][2],acc[i][j][3],
                             frA[p][i][0],frA[p][i][1],frA[p][i][2],frA[p][i][3],
                             frB[p][j][0],frB[p][j][1]);
            }
        }
    }

    const int g_epi=lane/4, t_epi=lane%4;
    for (int i=0;i<2;i++) for (int j=0;j<4;j++) {
        const int r0=by*T64_BM+wy*32+i*16+g_epi, r8=r0+8, col=bx*T64_BN+wx*32+j*8+t_epi*2;
        float* dC = C + (long long)batch * strideC;
        auto sf2=[&](int r,int c,float f0,float f1){
            if(r>=M||c>=N)return;
            float* dst=&dC[(long long)r*ldc+c];
            if constexpr(SplitK>1){atomicAdd(dst,alpha*f0);if(c+1<N)atomicAdd(dst+1,alpha*f1);}
            else{
                if(c+1<N&&(((size_t)dst&7)==0)){float2 old=(beta==0.f)?make_float2(0,0):*(float2*)dst;*(float2*)dst={alpha*f0+beta*old.x,alpha*f1+beta*old.y};}
                else{dst[0]=alpha*f0+(beta==0.f?0.f:beta*dst[0]);if(c+1<N)dst[1]=alpha*f1+(beta==0.f?0.f:beta*dst[1]);}
            }
        };
        sf2(r0,col,acc[i][j][0],acc[i][j][1]); sf2(r8,col,acc[i][j][2],acc[i][j][3]);
    }
}

// ============================================================
// TN  64x64 kernel  (C = A^T*B,  A[K,M] stored as M rows K cols, B[K,N])
// ============================================================
template <bool IsAligned, int SplitK>
__global__ void __launch_bounds__(T64_THREADS, 2)
sgemm_tn_tile64_kernel(
    int M, int N, int K, float alpha,
    const float* __restrict__ A, int lda, long long strideA,
    const float* __restrict__ B, int ldb, long long strideB,
    float beta,
    float* __restrict__ C, int ldc, long long strideC,
    int batchCount)
{
    const int batch  = blockIdx.z / SplitK;
    const int sk_idx = blockIdx.z % SplitK;
    if (batch >= batchCount) return;

    int bx = blockIdx.x, by = blockIdx.y;
    const int swizzle = 4;
    if (gridDim.y % swizzle == 0) {
        const int bi = blockIdx.y * gridDim.x + blockIdx.x;
        by = (bi % swizzle) + (bi / (gridDim.x * swizzle)) * swizzle;
        bx = (bi / swizzle) % gridDim.x;
    }
    if (by * T64_BM >= M || bx * T64_BN >= N) return;

    const int tid  = threadIdx.x;
    const int lane = tid & 31, wid = tid >> 5;
    const int wy   = wid >> 1, wx = wid & 1;

    const int k_tiles      = (K + T64_BK - 1) / T64_BK;
    const int tiles_per_sk = (k_tiles + SplitK - 1) / SplitK;
    const int k_start      = sk_idx * tiles_per_sk * T64_BK;
    const int k_end        = min(K, k_start + tiles_per_sk * T64_BK);

    const float* gA_ptr[4];
    const float* gB_ptr[2];
    for (int i = 0; i < 4; i++) {
        int r = (tid / 32) + i * 4;
        int c = (tid % 32) * 2;
        gA_ptr[i] = A + (long long)batch * strideA + (long long)(k_start + r) * lda + (by * T64_BM + c);
    }
    for (int i = 0; i < 2; i++) {
        int r = i * 8 + tid / 16;
        gB_ptr[i] = B + (long long)batch * strideB + (long long)(k_start + r) * ldb + (bx * T64_BN + (tid % 16) * 4);
    }

    extern __shared__ float smem[];
    float acc[2][4][4] = {};

    auto load_stage = [&](int stage, int ko) {
        float* As = smem + stage * T64_SS;
        float* Bs = As + T64_AS;
        for (int i = 0; i < 4; i++) {
            int r = (tid / 32) + i * 4, c = (tid % 32) * 2;
            int sc = c ^ ((r & 7) << 1);
            uint32_t sm_a = __cvta_generic_to_shared(&As[r * T64_BM + sc]);
            int gk = ko + r, gm = by * T64_BM + c;
            if constexpr (IsAligned) {
                int bytes = (gk < K && gm < M) ? max(0, min(8, (M - gm) * 4)) : 0;
                asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" :: "r"(sm_a), "l"(gA_ptr[i]), "r"(bytes));
            } else {
                float2 v = {0,0}; if (gk < K) { if (gm<M) v.x=gA_ptr[i][0]; if (gm+1<M) v.y=gA_ptr[i][1]; }
                *(float2*)&As[r * T64_BM + sc] = v;
            }
            gA_ptr[i] += T64_BK * lda;
        }
        for (int i = 0; i < 2; i++) {
            int r = i * 8 + tid / 16, c_base = (tid % 16) * 4;
            int sc = c_base ^ ((r & 7) << 2);
            uint32_t sm_b = __cvta_generic_to_shared(&Bs[r * T64_BN + sc]);
            int gr = ko + r, gc = bx * T64_BN + c_base;
            if constexpr (IsAligned) {
                int bytes = (gr < K) ? max(0, min(16, (N - gc) * 4)) : 0;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm_b), "l"(gB_ptr[i]), "r"(bytes));
            } else {
                float4 v={0,0,0,0}; if(gr<K){if(gc<N)v.x=gB_ptr[i][0];if(gc+1<N)v.y=gB_ptr[i][1];if(gc+2<N)v.z=gB_ptr[i][2];if(gc+3<N)v.w=gB_ptr[i][3];}
                *(float4*)&Bs[r * T64_BN + sc] = v;
            }
            gB_ptr[i] += (long long)T64_BK * ldb;
        }
    };

    const int g_sh = lane / 4, t_sh = lane % 4;
    int m_base[2]; for (int i=0;i<2;i++) m_base[i]=wy*32+i*16+g_sh;
    int n_base[4]; for (int j=0;j<4;j++) n_base[j]=wx*32+j*8+g_sh;

    auto load_frA = [&](uint32_t reg[4], int ks, int mi, int st) {
        float* As = smem + st * T64_SS;
        auto ga=[&](int k, int m){return *(uint32_t*)&As[k*T64_BM+(m^((k&7)<<1))];};
        reg[0]=ga(ks+t_sh,   m_base[mi]);   reg[1]=ga(ks+t_sh,   m_base[mi]+8);
        reg[2]=ga(ks+t_sh+4, m_base[mi]);   reg[3]=ga(ks+t_sh+4, m_base[mi]+8);
    };
    auto load_frB = [&](uint32_t reg[2], int ks, int ni, int st) {
        float* Bs = smem + st * T64_SS + T64_AS;
        auto gb=[&](int r, int c){return *(uint32_t*)&Bs[r*T64_BN+(c^((r&7)<<2))];};
        reg[0]=gb(ks+t_sh,   n_base[ni]); reg[1]=gb(ks+t_sh+4, n_base[ni]);
    };

    if (k_start < k_end) {
        load_stage(0, k_start); asm volatile("cp.async.commit_group;\n");
        if (k_start + T64_BK < k_end) { load_stage(1, k_start + T64_BK); asm volatile("cp.async.commit_group;\n"); }
    }
    int ws=2, rs=0;
    uint32_t frA[2][2][4], frB[2][4][2];
    asm volatile("cp.async.wait_group 1;\n"); __syncthreads();
    for (int i=0;i<2;i++) load_frA(frA[0][i],0,i,rs);
    for (int j=0;j<4;j++) load_frB(frB[0][j],0,j,rs);

    for (int k = k_start; k < k_end; k += T64_BK) {
        if (k+2*T64_BK<k_end) { load_stage(ws,k+2*T64_BK); asm volatile("cp.async.commit_group;\n"); }
        #pragma unroll
        for (int ks=0;ks<T64_BK;ks+=8){
            const int p=(ks/8)&1,q=p^1;
            #pragma unroll
            for (int i=0;i<2;i++){
                if(ks+8<T64_BK){load_frA(frA[q][i],ks+8,i,rs);load_frB(frB[q][i*2],ks+8,i*2,rs);load_frB(frB[q][i*2+1],ks+8,i*2+1,rs);}
                else if(k+T64_BK<k_end){if(i==0){asm volatile("cp.async.wait_group 1;\n");__syncthreads();rs=(rs+1)%T64_STAGES;ws=(ws+1)%T64_STAGES;}
                    load_frA(frA[q][i],0,i,rs);load_frB(frB[q][i*2],0,i*2,rs);load_frB(frB[q][i*2+1],0,i*2+1,rs);}
                #pragma unroll
                for (int j=0;j<4;j++)
                    MMA_TF32(acc[i][j][0],acc[i][j][1],acc[i][j][2],acc[i][j][3],
                             frA[p][i][0],frA[p][i][1],frA[p][i][2],frA[p][i][3],
                             frB[p][j][0],frB[p][j][1]);
            }
        }
    }

    const int g_epi=lane/4, t_epi=lane%4;
    for (int i=0;i<2;i++) for (int j=0;j<4;j++){
        const int r0=by*T64_BM+wy*32+i*16+g_epi,r8=r0+8,col=bx*T64_BN+wx*32+j*8+t_epi*2;
        float* dC=C+(long long)batch*strideC;
        auto sf2=[&](int r,int c,float f0,float f1){
            if(r>=M||c>=N)return; float* dst=&dC[(long long)r*ldc+c];
            if constexpr(SplitK>1){atomicAdd(dst,alpha*f0);if(c+1<N)atomicAdd(dst+1,alpha*f1);}
            else{if(c+1<N&&(((size_t)dst&7)==0)){float2 old=(beta==0.f)?make_float2(0,0):*(float2*)dst;*(float2*)dst={alpha*f0+beta*old.x,alpha*f1+beta*old.y};}
                 else{dst[0]=alpha*f0+(beta==0.f?0.f:beta*dst[0]);if(c+1<N)dst[1]=alpha*f1+(beta==0.f?0.f:beta*dst[1]);}}
        };
        sf2(r0,col,acc[i][j][0],acc[i][j][1]); sf2(r8,col,acc[i][j][2],acc[i][j][3]);
    }
}

__global__ void sgemm_tile64_scale_kernel(
    float* C, float beta, int M, int N, int ldc, long long strideC, int batchCount)
{
    int r=blockIdx.y*blockDim.y+threadIdx.y, c=blockIdx.x*blockDim.x+threadIdx.x, b=blockIdx.z;
    if (r<M&&c<N&&b<batchCount) {
        float* dst=&C[b*strideC+(long long)r*ldc+c];
        *dst=(beta==0.f)?0.f:(*dst*beta);
    }
}

static const size_t T64_SMEM = T64_STAGES * T64_SS * sizeof(float);

static void t64_set_limit(const void* f) {
    static thread_local std::unordered_map<const void*, bool> done;
    if (!done[f]) { cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)T64_SMEM); done[f]=true; }
}

#define T64_DISPATCH_NN(aligned, sk) \
    t64_set_limit((const void*)sgemm_nn_tile64_kernel<aligned, sk>); \
    sgemm_nn_tile64_kernel<aligned, sk><<<grid, T64_THREADS, T64_SMEM, stream>>>(M,N,K,alpha,d_A,lda,strideA,d_B,ldb,strideB,_beta,d_C,ldc,strideC,batchCount)

#define T64_DISPATCH_NT(aligned, sk) \
    t64_set_limit((const void*)sgemm_nt_tile64_kernel<aligned, sk>); \
    sgemm_nt_tile64_kernel<aligned, sk><<<grid, T64_THREADS, T64_SMEM, stream>>>(M,N,K,alpha,d_A,lda,strideA,d_B,ldb,strideB,_beta,d_C,ldc,strideC,batchCount)

#define T64_DISPATCH_TN(aligned, sk) \
    t64_set_limit((const void*)sgemm_tn_tile64_kernel<aligned, sk>); \
    sgemm_tn_tile64_kernel<aligned, sk><<<grid, T64_THREADS, T64_SMEM, stream>>>(M,N,K,alpha,d_A,lda,strideA,d_B,ldb,strideB,_beta,d_C,ldc,strideC,batchCount)

static int tile64_sk(int gx, int gy, int K, int sm_count) {
    if (K < 512) return 1;
    int sk = 4;
    int total_blocks = gx * gy;
    if (total_blocks < sm_count / 4) sk = 8;
    if (total_blocks < sm_count / 16) sk = 16;
    if (total_blocks == 1 && K >= 2048) sk = 32;
    return sk;
}

extern "C" void mycublasSgemmStridedBatched_nn_tile64_SM86(
    mycublasHandle_t handle, int M, int N, int K, float alpha,
    const float* d_A, int lda, long long strideA,
    const float* d_B, int ldb, long long strideB,
    float beta, float* d_C, int ldc, long long strideC, int batchCount)
{
    cudaStream_t stream = handle ? handle->stream : 0;
    const bool aligned = ((size_t)d_A&15)==0 && (lda&3)==0 && ((size_t)d_B&15)==0 && (ldb&3)==0;
    const int gx = (N + T64_BN - 1) / T64_BN;
    const int gy = (M + T64_BM - 1) / T64_BM;
    int sm_version, sm_count; get_gpu_info(&sm_version, &sm_count);
    int sk = tile64_sk(gx, gy, K, sm_count);
    float _beta = (sk > 1) ? 0.f : beta;
    if (sk > 1) {
        dim3 sg((N+31)/32,(M+31)/32,batchCount);
        sgemm_tile64_scale_kernel<<<sg,dim3(32,32),0,stream>>>(d_C,beta,M,N,ldc,strideC,batchCount);
    }
    dim3 grid(gx, gy, batchCount * sk);
    if (aligned) { if(sk==1){T64_DISPATCH_NN(true,1);}else if(sk==4){T64_DISPATCH_NN(true,4);}else if(sk==8){T64_DISPATCH_NN(true,8);}else if(sk==16){T64_DISPATCH_NN(true,16);}else{T64_DISPATCH_NN(true,32);} }
    else         { if(sk==1){T64_DISPATCH_NN(false,1);}else if(sk==4){T64_DISPATCH_NN(false,4);}else if(sk==8){T64_DISPATCH_NN(false,8);}else if(sk==16){T64_DISPATCH_NN(false,16);}else{T64_DISPATCH_NN(false,32);} }
}

extern "C" void mycublasSgemmStridedBatched_nt_tile64_SM86(
    mycublasHandle_t handle, int M, int N, int K, float alpha,
    const float* d_A, int lda, long long strideA,
    const float* d_B, int ldb, long long strideB,
    float beta, float* d_C, int ldc, long long strideC, int batchCount)
{
    cudaStream_t stream = handle ? handle->stream : 0;
    const bool aligned = ((size_t)d_A&15)==0 && (lda&3)==0 && ((size_t)d_B&15)==0 && (ldb&3)==0;
    const int gx = (N + T64_BN - 1) / T64_BN;
    const int gy = (M + T64_BM - 1) / T64_BM;
    int sm_version, sm_count; get_gpu_info(&sm_version, &sm_count);
    int sk = tile64_sk(gx, gy, K, sm_count);
    float _beta = (sk > 1) ? 0.f : beta;
    if (sk > 1) {
        dim3 sg((N+31)/32,(M+31)/32,batchCount);
        sgemm_tile64_scale_kernel<<<sg,dim3(32,32),0,stream>>>(d_C,beta,M,N,ldc,strideC,batchCount);
    }
    dim3 grid(gx, gy, batchCount * sk);
    if (aligned) { if(sk==1){T64_DISPATCH_NT(true,1);}else if(sk==4){T64_DISPATCH_NT(true,4);}else if(sk==8){T64_DISPATCH_NT(true,8);}else if(sk==16){T64_DISPATCH_NT(true,16);}else{T64_DISPATCH_NT(true,32);} }
    else         { if(sk==1){T64_DISPATCH_NT(false,1);}else if(sk==4){T64_DISPATCH_NT(false,4);}else if(sk==8){T64_DISPATCH_NT(false,8);}else if(sk==16){T64_DISPATCH_NT(false,16);}else{T64_DISPATCH_NT(false,32);} }
}

extern "C" void mycublasSgemmStridedBatched_tn_tile64_SM86(
    mycublasHandle_t handle, int M, int N, int K, float alpha,
    const float* d_A, int lda, long long strideA,
    const float* d_B, int ldb, long long strideB,
    float beta, float* d_C, int ldc, long long strideC, int batchCount)
{
    cudaStream_t stream = handle ? handle->stream : 0;
    const bool aligned = ((size_t)d_A&15)==0 && (lda&3)==0 && ((size_t)d_B&15)==0 && (ldb&3)==0;
    const int gx = (N + T64_BN - 1) / T64_BN;
    const int gy = (M + T64_BM - 1) / T64_BM;
    int sm_version, sm_count; get_gpu_info(&sm_version, &sm_count);
    int sk = tile64_sk(gx, gy, K, sm_count);
    float _beta = (sk > 1) ? 0.f : beta;
    if (sk > 1) {
        dim3 sg((N+31)/32,(M+31)/32,batchCount);
        sgemm_tile64_scale_kernel<<<sg,dim3(32,32),0,stream>>>(d_C,beta,M,N,ldc,strideC,batchCount);
    }
    dim3 grid(gx, gy, batchCount * sk);
    if (aligned) { if(sk==1){T64_DISPATCH_TN(true,1);}else if(sk==4){T64_DISPATCH_TN(true,4);}else if(sk==8){T64_DISPATCH_TN(true,8);}else if(sk==16){T64_DISPATCH_TN(true,16);}else{T64_DISPATCH_TN(true,32);} }
    else         { if(sk==1){T64_DISPATCH_TN(false,1);}else if(sk==4){T64_DISPATCH_TN(false,4);}else if(sk==8){T64_DISPATCH_TN(false,8);}else if(sk==16){T64_DISPATCH_TN(false,16);}else{T64_DISPATCH_TN(false,32);} }
}
