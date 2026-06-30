#include "mycublas.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>
#include <unordered_map>

using namespace nvcuda;

// ============================================================
// Hgemm Backward NT V18 — dA = dY dX^T (C = A * B^T)
// ============================================================

#define BM      128
#define BN      128
#define BK      32
#define STAGES  3
#define THREADS 128
#define AS_SIZE (BM * BK)
#define BS_SIZE (BN * BK)
#define STG_SZ  (AS_SIZE + BS_SIZE)

#define MMA_M16N8K16_F32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1) \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 " \
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};" \
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) \
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

template <bool Aligned>
__global__ void __launch_bounds__(THREADS, 1)
hgemm_nt_backward_kernel(
    int M, int N, int K,
    __half alpha,
    const __half* __restrict__ A, int lda, long long sA,
    const __half* __restrict__ B, int ldb, long long sB,
    __half beta,
    __half* __restrict__ C, int ldc, long long sC,
    int batchCount)
{
    const int batch = blockIdx.z;
    if (batch >= batchCount) return;

    int bx = blockIdx.x, by = blockIdx.y;
    const int sw = 8;
    if (gridDim.y % sw == 0) {
        const int bid = blockIdx.y * gridDim.x + blockIdx.x;
        by = (bid % sw) + (bid / (gridDim.x * sw)) * sw;
        bx = (bid / sw) % gridDim.x;
    }
    if (by * BM >= M || bx * BN >= N) return;

    const int tid = threadIdx.x;
    const int lane = tid & 31, wid = tid >> 5;
    const int wy = wid >> 1, wx = wid & 1;

    const __half* A_batch = A + (long long)batch * sA;
    const __half* B_batch = B + (long long)batch * sB;
    __half*       C_batch = C + (long long)batch * sC;

    extern __shared__ __half s_mem[];
    
    float acc[4][8][4];
    #pragma unroll
    for (int i = 0; i < 4; i++) for (int j = 0; j < 8; j++) for (int d = 0; d < 4; d++) acc[i][j][d] = 0.f;

    auto load_tile = [&](int stage, int ko) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int r = tid; int c = i * 8;
            const __half* g_ptr = A_batch + (long long)(by * BM + r) * lda + (ko + c);
            int swizzled_c = c ^ ((r & 3) << 3); 
            uint32_t sm_addr = __cvta_generic_to_shared(&s_mem[stage * STG_SZ + r * BK + swizzled_c]);
            int bytes = (by * BM + r < M && ko + c < K) ? max(0, min(16, (K - (ko + c)) * 2)) : 0;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm_addr), "l"(g_ptr), "r"(bytes));
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int r = tid; int c = i * 8;
            const __half* g_ptr = B_batch + (long long)(bx * BN + r) * ldb + (ko + c); 
            int swizzled_c = c ^ ((r & 3) << 3); 
            uint32_t sm_addr = __cvta_generic_to_shared(&s_mem[stage * STG_SZ + AS_SIZE + r * BK + swizzled_c]);
            int bytes = (bx * BN + r < N && ko + c < K) ? max(0, min(16, (K - (ko + c)) * 2)) : 0;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm_addr), "l"(g_ptr), "r"(bytes));
        }
    };

    load_tile(0, 0); asm volatile("cp.async.commit_group;\n");
    if (BK < K) { load_tile(1, BK); asm volatile("cp.async.commit_group;\n"); }
    
    int ws = 2, rs = 0;
    uint32_t frA[2][4][4], frB[2][8][2];

    asm volatile("cp.async.wait_group 1;\n"); __syncthreads();

    auto load_regA = [&](uint32_t reg[4], int ki, int mi, int st) {
        int r = wy * 64 + mi * 16 + ((lane / 8) % 2) * 8 + (lane % 8);
        int c = ki + (lane / 16) * 8;
        int swizzled_c = c ^ ((r & 3) << 3);
        uint32_t addr = __cvta_generic_to_shared(&s_mem[st * STG_SZ + r * BK + swizzled_c]);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];" 
                     : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3]) : "r"(addr));
    };

    auto load_regB = [&](uint32_t reg[2], int ki, int ni, int st) {
        int r = wx * 64 + ni * 16 + ((lane / 8) % 2) * 8 + (lane % 8);
        int c = ki + (lane / 16) * 8;
        int swizzled_c = c ^ ((r & 3) << 3);
        uint32_t addr = __cvta_generic_to_shared(&s_mem[st * STG_SZ + AS_SIZE + r * BK + swizzled_c]);
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];" 
                     : "=r"(reg[0]), "=r"(reg[1]) : "r"(addr));
    };

    #pragma unroll
    for (int i = 0; i < 4; i++) load_regA(frA[0][i], 0, i, rs);
    #pragma unroll
    for (int j = 0; j < 8; j++) load_regB(frB[0][j], 0, j, rs);

    for (int k = 0; k < K; k += BK) {
        if (k + 2 * BK < K) load_tile(ws, k + 2 * BK); 
        asm volatile("cp.async.commit_group;\n");

        #pragma unroll
        for (int ks = 0; ks < 32; ks += 16) {
            int p = (ks >> 4) & 1, q = p ^ 1;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                if (ks + 16 < 32) {
                    load_regA(frA[q][i], ks + 16, i, rs);
                    load_regB(frB[q][i*2], ks + 16, i*2, rs); load_regB(frB[q][i*2+1], ks + 16, i*2+1, rs);
                } else if (k + BK < K) {
                    if (i == 0) {
                        asm volatile("cp.async.wait_group 1;\n"); __syncthreads();
                        rs = (rs + 1) % STAGES; ws = (ws + 1) % STAGES;
                    }
                    load_regA(frA[q][i], 0, i, rs);
                    load_regB(frB[q][i*2], 0, i*2, rs); load_regB(frB[q][i*2+1], 0, i*2+1, rs);
                }
                #pragma unroll
                for (int j = 0; j < 8; j++) MMA_M16N8K16_F32(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3], frA[p][i][0], frA[p][i][1], frA[p][i][2], frA[p][i][3], frB[p][j][0], frB[p][j][1]);
            }
        }
    }

    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    float alpha_f = __half2float(alpha);
    float beta_f = __half2float(beta);
    const int rt = lane / 4; 
    const int ct = (lane % 4) * 2; 

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int r = wy * 64 + i * 16 + rt;
            int c = wx * 64 + j * 8 + ct;
            s_mem[r * BN + c]     = __float2half(acc[i][j][0] * alpha_f);
            s_mem[r * BN + c + 1] = __float2half(acc[i][j][1] * alpha_f);
            s_mem[(r+8) * BN + c]     = __float2half(acc[i][j][2] * alpha_f);
            s_mem[(r+8) * BN + c + 1] = __float2half(acc[i][j][3] * alpha_f);
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int r = (tid / 16) + i * 8; 
        int c = (tid % 16) * 8;
        int gr = by * BM + r;
        int gc = bx * BN + c;

        if (gr < M && gc < N) {
            __half* d_ptr = C_batch + (long long)gr * ldc + gc;
            __half* s_ptr = &s_mem[r * BN + c];
            #pragma unroll
            for (int l = 0; l < 8; l++) {
                if (gc + l < N) {
                    d_ptr[l] = __float2half(__half2float(s_ptr[l]) + beta_f * __half2float(d_ptr[l]));
                }
            }
        }
    }
}

extern "C" void mycublasHgemmStridedBatched_nt(
    mycublasHandle_t handle, int M, int N, int K, const __half alpha,
    const __half* d_A, int lda, long long int strideA,
    const __half* d_B, int ldb, long long int strideB,
    const __half beta, __half* d_C, int ldc, long long int strideC, 
    int batchCount)
{
    cudaStream_t stream = (handle != nullptr) ? handle->stream : 0;
    static const size_t smem_bytes = STAGES * STG_SZ * sizeof(__half);
    const bool aligned = (((size_t)d_A & 15) == 0) && ((lda & 7) == 0) && (((size_t)d_B & 15) == 0) && ((ldb & 7) == 0);
    
    auto set_attr = [](const void* f, size_t b) { 
        static std::unordered_map<const void*, bool> done; 
        if (!done[f]) { cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)b); done[f] = true; } 
    };

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, batchCount);
    if (aligned) {
        set_attr((const void*)hgemm_nt_backward_kernel<true>, smem_bytes);
        hgemm_nt_backward_kernel<true><<<grid, THREADS, smem_bytes, stream>>>(M,N,K,alpha,d_A,lda,strideA,d_B,ldb,strideB,beta,d_C,ldc,strideC,batchCount);
    } else {
        set_attr((const void*)hgemm_nt_backward_kernel<false>, smem_bytes);
        hgemm_nt_backward_kernel<false><<<grid, THREADS, smem_bytes, stream>>>(M,N,K,alpha,d_A,lda,strideA,d_B,ldb,strideB,beta,d_C,ldc,strideC,batchCount);
    }
}
