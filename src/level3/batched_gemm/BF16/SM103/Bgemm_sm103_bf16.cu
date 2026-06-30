// Bgemm_sm103_bf16.cu
// Blackwell Ultra (SM103a / B300) BF16 Batched GEMM — NT layout
// C[b,M,N] = alpha * A[b,M,K] × B[b,N,K]^T + beta * C[b,M,N]
//
// Architecture:
//   tcgen05.mma.cta_group::1.kind::f16  (SM103a supports cta_group::1, NOT cta_group::2)
//   TMA 4D bulk async loads with mbarrier pipeline
//   Single CTA per output tile (no CTA cluster — B300 has no cta_group::2 support)
//   Warp roles: 4 epilogue | 1 producer | 1 consumer
//
// Differences vs SM100 kernel (Bgemm_sm100_bf16.cu):
//   • CTA_GROUP_SIZE = 1  (each CTA owns the full BM × BN tile)
//   • cta_group::1 for all tcgen05 operations
//   • No __cluster_dims__, no cluster sync, no mapa
//   • B TMA loads full BN rows (not BN/2)
//   • i_desc M-bits encode BM (not 2*BM)
//
// Tile: BM=128, BN=128, BK=64

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "mycublas.h"

// ─── compile-time constants ───────────────────────────────────────────────────

static constexpr int WARP_SIZE      = 32;
static constexpr int SWIZZLE_W      = 64;
static constexpr int MMA_K_STEP     = 16;
// static constexpr int CTA_GROUP_SIZE = 1;    // single CTA per tile (unused)
static constexpr int NUM_EPI_WARPS  = 4;
static constexpr int NUM_PROD_WARPS = 1;
static constexpr int NUM_CONS_WARPS = 1;
static constexpr int TOTAL_WARPS    = NUM_EPI_WARPS + NUM_PROD_WARPS + NUM_CONS_WARPS;
static constexpr int STORE_N        = 64;
static constexpr int NUM_STORE_STGS = 2;



// ─── device primitives ───────────────────────────────────────────────────────

__device__ __forceinline__
uint32_t elect_sync()
{
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}"
        : "+r"(pred) : "r"(0xFFFFFFFFu));
    return pred;
}

__device__ __forceinline__
void mbarrier_init(int addr, int count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :: "r"(addr), "r"(count));
}

__device__ __forceinline__
void mbarrier_arrive_expect(int addr, int tx)
{
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(addr), "r"(tx) : "memory");
}

__device__ __forceinline__
void mbarrier_wait(int addr, int phase)
{
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%0], %1, %2;\n\t"
        "@P bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}"
        :: "r"(addr), "r"(phase), "r"(ticks));
}

// TMA load: 4D bulk async into smem (single CTA — no multicast).
__device__ __forceinline__
void tma_4d_gmem2smem(int dst, const void* tmap,
                      int x, int y, int z, int w, int mbar)
{
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap),
           "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar)
        : "memory");
}

__device__ __forceinline__
constexpr uint64_t desc_encode(uint64_t x)
{
    return (x & 0x3'FFFFULL) >> 4ULL;
}

__device__ __forceinline__
uint64_t make_smem_desc(int addr, int leading_dim_offset = 0)
{
    const int stride = 8 * SWIZZLE_W * static_cast<int>(sizeof(__nv_bfloat16));
    return desc_encode(addr)
         | (desc_encode(leading_dim_offset) << 16ULL)
         | (desc_encode(stride) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
}

// tcgen05.mma.cta_group::1 — single CTA accumulation.
__device__ __forceinline__
void tcgen05_mma_bf16_1cta(int tmem_addr,
                            uint64_t a_desc, uint64_t b_desc,
                            uint32_t i_desc, int use_accum)
{
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
        "}"
        :: "r"(tmem_addr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(use_accum));
}

__device__ __forceinline__
void alloc_tmem_1cta(int tmem_smem_addr, int width)
{
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(tmem_smem_addr), "r"(width));
}

__device__ __forceinline__
void dealloc_tmem_1cta(int tmem_addr, int width)
{
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        :: "r"(tmem_addr), "r"(width));
}

// Plain mbarrier arrive — for signalling smem stage is free (empty_mbar).
// Does NOT commit TMEM. Use tcgen05_commit_1cta for TMEM-full signals.
__device__ __forceinline__
void mbar_arrive_1cta(int mbar_addr)
{
    asm volatile(
        "mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr)
        : "memory");
}

// tcgen05 TMEM commit + mbarrier arrive — for signalling TMEM is fully written.
// shared::cluster scope is required even in a single-CTA cluster (per PTX ISA).
// Reference: CUTLASS cutlass/arch/barrier.h::umma_arrive()
__device__ __forceinline__
void tcgen05_commit_1cta(int mbar_addr)
{
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
        :: "r"(mbar_addr)
        : "memory");
}

__device__ __forceinline__
void tcgen05_ld(float* tmp, int addr)
{
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32"
        " {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        : "=f"(tmp[0]),"=f"(tmp[1]),"=f"(tmp[2]),"=f"(tmp[3]),
          "=f"(tmp[4]),"=f"(tmp[5]),"=f"(tmp[6]),"=f"(tmp[7])
        : "r"(addr));
}

__device__ __forceinline__
void tcgen05_wait_ld()
{
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}

__device__ __forceinline__
void tcgen05_fence_before()
{
    asm volatile("tcgen05.fence::before_thread_sync;");
}

__device__ __forceinline__
void tcgen05_fence_after()
{
    asm volatile("tcgen05.fence::after_thread_sync;");
}

__device__ __forceinline__
void tma_store_fence()
{
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__
void tma_store_commit()
{
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

template<int N>
__device__ __forceinline__
void tma_store_wait()
{
    asm volatile("cp.async.bulk.wait_group %0;" :: "n"(N) : "memory");
}

__device__ __forceinline__
void tma_2d_smem2gmem(int src, const void* tmap, int x, int y)
{
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
        " [%0, {%2, %3}], [%1];"
        :: "l"(tmap), "r"(src), "r"(x), "r"(y)
        : "memory");
}

__device__ __forceinline__
void barrier_sync(int bar, int n)
{
    asm volatile("bar.sync %0, %1;" :: "r"(bar), "r"(n));
}

// ─── host-side TMA map helpers ────────────────────────────────────────────────

static inline void check_cu(CUresult e)
{
    if (e != CUDA_SUCCESS)
    {
        const char* msg = nullptr;
        cuGetErrorString(e, &msg);
        fprintf(stderr, "[TMA Error] cuTensorMapEncodeTiled failed: %s (%d)\n", msg ? msg : "unknown", (int)e);
        exit(EXIT_FAILURE);
    }
}

// 4D TMA for A or B.
// A[batch, M, K]:      globalDim = {64, M,   K/64, batch},   box = {64, BM, BK/64, 1}
// B[batch, N, K]:      globalDim = {64, N,   K/64, batch},   box = {64, BN, BK/64, 1}
static inline void init_4d_tma_map_sm103(
    CUtensorMap* tmap,
    const __nv_bfloat16* ptr,
    int K, int BK,
    uint64_t global_height,   // M for A, N for B (full N — no CTA splitting)
    uint32_t box_height,      // BM for A, BN for B
    uint64_t batch_count,
    uint64_t batch_stride_elems,
    CUtensorMapSwizzle swizzle)
{
    constexpr uint32_t rank = 4;
    uint64_t globalDim[rank]      = {64ULL, global_height,
                                      (uint64_t)(K / 64), batch_count};
    uint64_t globalStrides[rank-1] = {
        (uint64_t)K * sizeof(__nv_bfloat16),
        128ULL,
        batch_stride_elems * sizeof(__nv_bfloat16)
    };
    uint32_t boxDim[rank]         = {64U, box_height, (uint32_t)(BK / 64), 1U};
    uint32_t elementStrides[rank] = {1U, 1U, 1U, 1U};

    check_cu(cuTensorMapEncodeTiled(
        tmap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        rank, (void*)ptr,
        globalDim, globalStrides, boxDim, elementStrides,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

static inline void init_2d_tma_map_store_sm103(
    CUtensorMap* tmap,
    const __nv_bfloat16* ptr,
    uint64_t total_rows, uint64_t total_cols,
    uint32_t box_rows,   uint32_t box_cols,
    CUtensorMapSwizzle swizzle)
{
    constexpr uint32_t rank = 2;
    uint64_t globalDim[rank]      = {total_cols, total_rows};
    uint64_t globalStrides[rank-1] = {total_cols * sizeof(__nv_bfloat16)};
    uint32_t boxDim[rank]         = {box_cols, box_rows};
    uint32_t elementStrides[rank] = {1U, 1U};

    check_cu(cuTensorMapEncodeTiled(
        tmap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        rank, (void*)ptr,
        globalDim, globalStrides, boxDim, elementStrides,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

// ─── kernel ───────────────────────────────────────────────────────────────────
//
// Grid: dim3(ceil(N/BN), ceil(M/BM), batchCount)
// Block: TOTAL_WARPS * WARP_SIZE = 192 threads
// No cluster — each CTA independently accumulates BM × BN.
//
// Warp assignments:
//   0-3 : epilogue  — tcgen05.ld → alpha/beta → TMA store
//   4   : producer  — TMA 4D loads for A and B tiles
//   5   : consumer  — tcgen05.mma.cta_group::1 across K

template<int BM, int BN, int BK, int QUEUE_SIZE>
__global__
void bgemm_sm103_bf16_nt_kernel(
    const __nv_bfloat16* __restrict__ A,
          __nv_bfloat16* __restrict__ /* unused raw ptr — accessed via TMA */,
          __nv_bfloat16* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta,
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B_tmap,
    const __grid_constant__ CUtensorMap C_tmap)
{
    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int batch   = blockIdx.z;
    const int block_m = blockIdx.y;
    const int block_n = blockIdx.x;

    // ── shared memory layout ──────────────────────────────────────────────────
    // [QUEUE_SIZE × (BM×BK + BN×BK)] pipeline stages + store double-buffer
    extern __shared__ __align__(1024) char smem[];
    const int smem_ptr = static_cast<int>(__cvta_generic_to_shared(smem));

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ int tmem[1];
    const int tmem_smem_addr = static_cast<int>(__cvta_generic_to_shared(tmem));

    __shared__ __align__(8) uint64_t full_mbar[QUEUE_SIZE];
    __shared__ __align__(8) uint64_t empty_mbar[QUEUE_SIZE];
    __shared__ __align__(8) uint64_t tmem_full;
    __shared__ __align__(8) uint64_t tmem_empty;

    const int full_mbar_addr  = static_cast<int>(__cvta_generic_to_shared(full_mbar));
    const int empty_mbar_addr = static_cast<int>(__cvta_generic_to_shared(empty_mbar));
    const int tmem_full_addr  = static_cast<int>(__cvta_generic_to_shared(&tmem_full));
    const int tmem_empty_addr = static_cast<int>(__cvta_generic_to_shared(&tmem_empty));

    constexpr int producer_warp = NUM_EPI_WARPS;
    constexpr int consumer_warp = NUM_EPI_WARPS + 1;

    // Smem per stage: A tile + B tile (full BN — no cluster split)
    constexpr int a_tile_bytes = BM * BK * (int)sizeof(__nv_bfloat16);
    constexpr int b_tile_bytes = BN * BK * (int)sizeof(__nv_bfloat16);
    constexpr int copy_size    = a_tile_bytes + b_tile_bytes;
    constexpr int store_buf_sz = BM * STORE_N * (int)sizeof(__nv_bfloat16);
    const int store_smem       = smem_ptr + QUEUE_SIZE * copy_size;
    __nv_bfloat16* store_base  = reinterpret_cast<__nv_bfloat16*>(smem + QUEUE_SIZE * copy_size);

    // ── initialisation ────────────────────────────────────────────────────────
    if (warp_id == producer_warp && elect_sync())
    {
        for (int i = 0; i < QUEUE_SIZE; ++i)
        {
            // full_mbar: 1 producer arrival per stage
            mbarrier_init(full_mbar_addr  + i * 8, 1);
            // empty_mbar: 1 consumer arrival per stage
            mbarrier_init(empty_mbar_addr + i * 8, 1);
        }
        mbarrier_init(tmem_full_addr,  1);                    // consumer → epilogue
        mbarrier_init(tmem_empty_addr, NUM_EPI_WARPS);        // epilogue → (unused here; future)
    }
    else if (warp_id == consumer_warp)
    {
        alloc_tmem_1cta(tmem_smem_addr, BN);
    }
    __syncthreads();

    const int tmem_addr = tmem[0];

    // i_desc for cta_group::1: M=BM (single CTA), N=BN
    constexpr uint32_t i_desc =
          (1U <<  4U)
        | (1U <<  7U)
        | (1U << 10U)
        | ((uint32_t)BN  >> 3U << 17U)
        | ((uint32_t)BM  >> 4U << 24U);

    const int k_iters = K / BK;

    // ── producer warp ─────────────────────────────────────────────────────────
    if (warp_id == producer_warp && elect_sync())
    {
        const int A_row = block_m * BM;
        const int B_row = block_n * BN;    // full BN — no cluster split

        int stage = 0, phase = 0, issued = 0;

        for (int k = 0; k < k_iters; ++k)
        {
            if (issued >= QUEUE_SIZE)
                mbarrier_wait(empty_mbar_addr + stage * 8, phase ^ 1);
            ++issued;

            const int A_smem = smem_ptr + stage * copy_size;
            const int B_smem = A_smem + a_tile_bytes;
            const int mbar   = full_mbar_addr + stage * 8;

            mbarrier_arrive_expect(mbar, copy_size);

            tma_4d_gmem2smem(
                A_smem, &A_tmap,
                0, A_row, k * BK / SWIZZLE_W, batch, mbar);
            tma_4d_gmem2smem(
                B_smem, &B_tmap,
                0, B_row, k * BK / SWIZZLE_W, batch, mbar);

            stage = (stage + 1) % QUEUE_SIZE;
            phase ^= (stage == 0);
        }
    }
    // ── consumer warp ─────────────────────────────────────────────────────────
    else if (warp_id == consumer_warp && elect_sync())
    {
        asm volatile("tcgen05.fence::after_thread_sync;");

        int stage = 0, phase = 0;

        for (int k = 0; k < k_iters; ++k)
        {
            mbarrier_wait(full_mbar_addr + stage * 8, phase);
            asm volatile("tcgen05.fence::after_thread_sync;");

            const int A_smem = smem_ptr + stage * copy_size;
            const int B_smem = A_smem + a_tile_bytes;

            for (int k1 = 0; k1 < BK / SWIZZLE_W; ++k1)
            {
                for (int k2 = 0; k2 < SWIZZLE_W / MMA_K_STEP; ++k2)
                {
                    const int use_accum = (k == 0 && k1 == 0 && k2 == 0) ? 0 : 1;

                    const int a_off = k1 * SWIZZLE_W * BM * (int)sizeof(__nv_bfloat16)
                                    + k2 * MMA_K_STEP  * (int)sizeof(__nv_bfloat16);
                    const int b_off = k1 * SWIZZLE_W * BN * (int)sizeof(__nv_bfloat16)
                                    + k2 * MMA_K_STEP  * (int)sizeof(__nv_bfloat16);

                    tcgen05_mma_bf16_1cta(
                        tmem_addr,
                        make_smem_desc(A_smem + a_off),
                        make_smem_desc(B_smem + b_off),
                        i_desc, use_accum);
                }
            }

            // Free this smem stage (signal producer).
            {
                mbar_arrive_1cta(empty_mbar_addr + stage * 8);
            }

            stage = (stage + 1) % QUEUE_SIZE;
            phase ^= (stage == 0);
        }

        // Commit all TMEM writes and signal epilogue.
        // tcgen05.commit waits for outstanding TMEM writes and atomically signals
        // the mbarrier — plain mbarrier.arrive would NOT commit TMEM.
        asm volatile("tcgen05.fence::before_thread_sync;");
        tcgen05_commit_1cta(tmem_full_addr);
    }
    // ── epilogue warps ────────────────────────────────────────────────────────
    else if (warp_id < NUM_EPI_WARPS)
    {
        constexpr int EPI_BAR     = 7;
        constexpr int EPI_THREADS = NUM_EPI_WARPS * WARP_SIZE;
        constexpr int num_chunks  = BN / STORE_N;     // 2 chunks of 64 columns
        constexpr int LOADS_PER   = STORE_N / 8;

        mbarrier_wait(tmem_full_addr, 0);
        tcgen05_fence_after();

        // Each warp handles 32 rows of the BM × BN output tile.
        // High 16 bits of TMEM addr = row offset within this CTA's TMEM region.
        const int tmem_row = tmem_addr + (warp_id * WARP_SIZE << 16);
        const int row      = warp_id * WARP_SIZE + lane_id;
        int store_stage    = 0;

        for (int chunk = 0; chunk < num_chunks; ++chunk)
        {
            if (warp_id == 0) tma_store_wait<NUM_STORE_STGS - 1>();

            float tmp[LOADS_PER][8];
            #pragma unroll
            for (int n = 0; n < LOADS_PER; ++n)
                tcgen05_ld(tmp[n], tmem_row + chunk * STORE_N + n * 8);
            tcgen05_wait_ld();

            if (chunk == num_chunks - 1)
                tcgen05_fence_before();

            barrier_sync(EPI_BAR, EPI_THREADS);

            #pragma unroll
            for (int n = 0; n < LOADS_PER; ++n)
            {
                nv_bfloat162 packed[4];
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    float v0 = tmp[n][i * 2]     * alpha;
                    float v1 = tmp[n][i * 2 + 1] * alpha;
                    packed[i] = __float22bfloat162_rn({v0, v1});
                }
                if (beta != 0.0f)
                {
                    const int c_col = block_n * BN + chunk * STORE_N + n * 8;
                    if (c_col < N)
                    {
                        const __nv_bfloat16* c_ptr =
                            C + (long long)(batch * M + block_m * BM + row) * N + c_col;
                        #pragma unroll
                        for (int i = 0; i < 4; ++i)
                        {
                            __nv_bfloat162 old = *reinterpret_cast<const __nv_bfloat162*>(c_ptr + i * 2);
                            packed[i] = __hadd2(packed[i],
                                         __hmul2(old, __float22bfloat162_rn({beta, beta})));
                        }
                    }
                }
                const int swizzled_n = n ^ (row & 7);
                __nv_bfloat16* wp = store_base
                                  + store_stage * BM * STORE_N
                                  + row * STORE_N + swizzled_n * 8;
                *reinterpret_cast<int4*>(wp) = *reinterpret_cast<int4*>(packed);
            }

            __syncwarp();
            tma_store_fence();
            barrier_sync(EPI_BAR, EPI_THREADS);

            if (warp_id == 0 && elect_sync())
            {
                const int src   = store_smem + store_stage * store_buf_sz;
                const int c_col = block_n * BN + chunk * STORE_N;
                const int c_row = batch * M + block_m * BM;
                tma_2d_smem2gmem(src, &C_tmap, c_col, c_row);
                tma_store_commit();
            }

            store_stage ^= 1;
        }

        if (warp_id == 0) tma_store_wait<0>();
        barrier_sync(EPI_BAR, EPI_THREADS);
    }

    __syncthreads();

    if (warp_id == consumer_warp && elect_sync())
        dealloc_tmem_1cta(tmem_addr, BN);
}

// ─── launcher ─────────────────────────────────────────────────────────────────

template<int BM, int BN, int BK>
static void launch_bgemm_sm103_bf16_nt(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
          __nv_bfloat16* C,
    int M, int N, int K, int batchCount,
    float alpha, float beta,
    cudaStream_t stream)
{
    // Runtime arch check — cached for the process lifetime.
    static int s_arch_ok = -1;
    if (s_arch_ok < 0) {
        int dev = 0; cudaGetDevice(&dev);
        cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
        s_arch_ok = (prop.major == 10 && prop.minor == 3) ? 1 : 0;
        if (!s_arch_ok)
            fprintf(stderr,
                "[BluBridge SM103 kernel] WARNING: device '%s' is SM %d.%d. "
                "Requires SM 10.3 (B300/sm_103a). Skipping.\n",
                prop.name, prop.major, prop.minor);
    }
    if (!s_arch_ok) return;

    CUtensorMap A_tmap, B_tmap, C_tmap;

    // A: [batch, M, K] row-major → 4D TMA
    init_4d_tma_map_sm103(
        &A_tmap, A, K, BK,
        (uint64_t)M, (uint32_t)BM,
        (uint64_t)batchCount, (uint64_t)M * K,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);

    // B: [batch, N, K] row-major — full BN rows per CTA (no cluster split)
    init_4d_tma_map_sm103(
        &B_tmap, B, K, BK,
        (uint64_t)N, (uint32_t)BN,
        (uint64_t)batchCount, (uint64_t)N * K,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);

    // C: [batch*M, N] for TMA store
    init_2d_tma_map_store_sm103(
        &C_tmap, C,
        (uint64_t)batchCount * M, (uint64_t)N,
        (uint32_t)BM, (uint32_t)STORE_N,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);

    // Smem budget (BN full, no CTA split)
    constexpr int a_tile    = BM * BK * (int)sizeof(__nv_bfloat16);
    constexpr int b_tile    = BN * BK * (int)sizeof(__nv_bfloat16);
    constexpr int tile_size = a_tile + b_tile;
    constexpr int store_tot = NUM_STORE_STGS * BM * STORE_N * (int)sizeof(__nv_bfloat16);
    constexpr int smem_budget = 227 * 1024;
    constexpr int QUEUE_SIZE  = (smem_budget - store_tot) / tile_size;
    static_assert(QUEUE_SIZE >= 2, "QUEUE_SIZE too small");
    constexpr int smem_size   = QUEUE_SIZE * tile_size + store_tot;

    const int grid_m = (M + BM - 1) / BM;
    const int grid_n = (N + BN - 1) / BN;
    dim3 grid(grid_n, grid_m, batchCount);
    const int block_size = TOTAL_WARPS * WARP_SIZE;

    auto kernel = bgemm_sm103_bf16_nt_kernel<BM, BN, BK, QUEUE_SIZE>;
    cudaError_t set_attr_err = cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (set_attr_err != cudaSuccess) {
        fprintf(stderr, "[BluBridge SM103 kernel] cudaFuncSetAttribute failed: %s\n",
                cudaGetErrorString(set_attr_err));
    }

    kernel<<<grid, block_size, smem_size, stream>>>(
        A, nullptr, C,
        M, N, K,
        alpha, beta,
        A_tmap, B_tmap, C_tmap);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "[BluBridge SM103 kernel] Launch error: %s\n",
                cudaGetErrorString(err));
}

// ─── public C API ─────────────────────────────────────────────────────────────

extern "C" void mycublasBgemmSM103_bf16_nt_128x128x64(
    mycublasHandle_t handle,
    int M, int N, int K,
    float alpha,
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    float beta,
          __nv_bfloat16* C,
    int batchCount)
{
    cudaStream_t stream = handle ? handle->stream : 0;
    launch_bgemm_sm103_bf16_nt<128, 128, 64>(
        A, B, C, M, N, K, batchCount, alpha, beta, stream);
}

extern "C" void mycublasBgemmSM103_bf16_nt_strided(
    mycublasHandle_t handle,
    int M, int N, int K,
    float alpha,
    const __nv_bfloat16* A, long long strideA,
    const __nv_bfloat16* B, long long strideB,
    float beta,
          __nv_bfloat16* C, long long strideC,
    int batchCount)
{
    (void)strideA; (void)strideB; (void)strideC;
    cudaStream_t stream = handle ? handle->stream : 0;
    launch_bgemm_sm103_bf16_nt<128, 128, 64>(
        A, B, C, M, N, K, batchCount, alpha, beta, stream);
}


