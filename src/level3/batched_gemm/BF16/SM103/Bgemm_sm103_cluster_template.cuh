#pragma once
// Bgemm_sm103_cluster_template.cuh
// Unified SM103a (B300 / Blackwell Ultra) BF16 batched GEMM kernel template.
//
// Usage — include this header and call the launcher:
//
//   using Cfg = Sm103ClusterConfig<128, 128, 64>;   // BM, BN, BK
//   launch_bgemm_sm103_cluster<Cfg, Sm103Layout::NN>(A, B, C, M, N, K, batch, alpha, beta, stream);
//
// Architecture overview (identical for all three layouts):
//   - tcgen05.mma.cta_group::2.kind::f16   (2-CTA cooperative MMA — SM103a / B300)
//   - CLC (Cluster Launch Control) persistent tile scheduling
//   - On-device tile coords: block_col = mn_tile % grid_n  (no GPU-side tile_map)
//   - Warp layout:  [0 .. NUM_EPI_WARPS)  epilogue
//                   [NUM_EPI_WARPS]       producer
//                   [NUM_EPI_WARPS+1]     consumer  (drives MMA from CTA0)
//                   [NUM_EPI_WARPS+2]     scheduler (CLC loop)
//
// Per-layout differences (controlled by Sm103Layout template tag):
//   NN  A[batch,M,K] × B[batch,K,N]      trans_b=1 in i_desc
//   NT  A[batch,M,K] × B[batch,N,K]ᵀ    no trans bits (B stored [N,K])
//   TN  A[batch,K,M]ᵀ × B[batch,K,N]   trans_a=1 + trans_b=1 in i_desc

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

// ─── layout tag ──────────────────────────────────────────────────────────────

enum class Sm103Layout { NN, NT, TN };

// ─── config struct ────────────────────────────────────────────────────────────
//
// Maps directly to ThunderKittens' `config` struct pattern.
//
// Template parameters:
//   _BM             — M-tile rows per CTA (must be multiple of 32; cluster covers 2×BM)
//   _BN             — N-tile cols per cluster (must be multiple of 128 = 2×SWIZZLE_W)
//   _BK             — K-depth per pipeline stage (must be multiple of 64 = SWIZZLE_W)
//   _LOAD_PIPE_DEPTH — smem A/B pipeline depth (= QUEUE_SIZE in original code)
//   _EPI_PIPE_DEPTH  — TMEM double-buffering slots (= NUM_EPI_STAGES, default 2)
//   _CLC_PIPE_DEPTH  — CLC pipeline depth (= NUM_CLC_STAGES, default 2)

template<int _BM, int _BN, int _BK,
         int _LOAD_PIPE_DEPTH = 0,   // 0 = auto-derive from smem budget
         int _EPI_PIPE_DEPTH  = 2,
         int _CLC_PIPE_DEPTH  = 2>
struct Sm103ClusterConfig {

    // ── hardware-fixed SM103a constants ──────────────────────────────────────
    static constexpr int WARP_THREADS   = 32;
    static constexpr int SWIZZLE_W      = 64;    // 64 bf16 = 128 bytes, one swizzle group
    static constexpr int MMA_K_STEP     = 16;    // inner-K step of one tcgen05 MMA call
    static constexpr int CLUSTER_SIZE   = 2;     // 2-CTA cooperative MMA cluster
    static constexpr int STORE_COLS     = 64;    // N-columns per TMA store call
    static constexpr int STORE_PIPE_DEPTH = 2;   // store smem double buffer (fixed)
    static constexpr uint16_t CTA_MASK  = 0b11;  // multicast mask for 2-CTA cluster

    // ── tile dimensions ──────────────────────────────────────────────────────
    static constexpr int BM = _BM;   // M-rows per CTA
    static constexpr int BN = _BN;   // N-cols per cluster (each CTA holds BN/2)
    static constexpr int BK = _BK;   // K-depth per pipeline stage

    // ── pipeline depths ──────────────────────────────────────────────────────
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;  // TMEM slots
    static constexpr int CLC_PIPE_DEPTH = _CLC_PIPE_DEPTH;  // CLC stages

    // ── warp layout ──────────────────────────────────────────────────────────
    static constexpr int NUM_EPI_WARPS  = BM / WARP_THREADS; // one per 32 M-rows
    static constexpr int NUM_PROD_WARPS = 1;
    static constexpr int NUM_CONS_WARPS = 1;
    static constexpr int NUM_SCHED_WARPS = 1;
    static constexpr int NUM_WARPS      = NUM_EPI_WARPS + NUM_PROD_WARPS
                                        + NUM_CONS_WARPS + NUM_SCHED_WARPS;
    static constexpr int NUM_THREADS    = NUM_WARPS * WARP_THREADS;

    // ── smem sizes (bytes) ───────────────────────────────────────────────────
    // A smem: BM × BK bf16 per stage (TN has [BK×BM] but same byte count)
    static constexpr int A_STAGE_BYTES  = BM * BK * 2;
    // B smem: (BN/2) × BK bf16 per stage (each CTA holds half the N columns)
    static constexpr int B_STAGE_BYTES  = (BN / CLUSTER_SIZE) * BK * 2;
    static constexpr int STAGE_BYTES    = A_STAGE_BYTES + B_STAGE_BYTES;
    // Store buffer: double-buffered STORE_COLS-wide tiles
    static constexpr int STORE_BUF_BYTES = STORE_PIPE_DEPTH * BM * STORE_COLS * 2;
    // Maximum usable smem on B300 per CTA
    static constexpr int SMEM_BUDGET    = 227 * 1024;

    // ── load pipeline depth ──────────────────────────────────────────────────
    // If the caller passes 0 (default), auto-derive the maximum depth that fits.
    // If the caller passes an explicit value it is used verbatim (asserted below).
    static constexpr int LOAD_PIPE_DEPTH =
        (_LOAD_PIPE_DEPTH == 0)
            ? (SMEM_BUDGET - STORE_BUF_BYTES) / STAGE_BYTES
            : _LOAD_PIPE_DEPTH;

    static constexpr int SMEM_BYTES = LOAD_PIPE_DEPTH * STAGE_BYTES + STORE_BUF_BYTES;

    // ── CLC barrier arrive count ─────────────────────────────────────────────
    // All warps in both CTAs arrive on the CLC-empty barrier each tile.
    static constexpr int CLC_EMPTY_ARRIVE =
        (NUM_EPI_WARPS + NUM_PROD_WARPS + NUM_CONS_WARPS + NUM_SCHED_WARPS) * CLUSTER_SIZE;

    // ── warp index offsets ───────────────────────────────────────────────────
    static constexpr int PRODUCER_WARP  = NUM_EPI_WARPS;
    static constexpr int CONSUMER_WARP  = NUM_EPI_WARPS + 1;
    static constexpr int SCHEDULER_WARP = NUM_EPI_WARPS + 2;

    // ── validity checks ──────────────────────────────────────────────────────
    static_assert(BM >= 32 && BM % WARP_THREADS == 0,
        "BM must be a positive multiple of 32 (one warp per row group)");
    static_assert(BN >= 128 && BN % (CLUSTER_SIZE * SWIZZLE_W) == 0,
        "BN must be a multiple of 128 (CLUSTER_SIZE × SWIZZLE_W); each CTA holds BN/2");
    static_assert(BK >= 64 && BK % SWIZZLE_W == 0,
        "BK must be a multiple of 64 (SWIZZLE_W)");
    static_assert(LOAD_PIPE_DEPTH >= 2,
        "LOAD_PIPE_DEPTH must be >= 2 (need at least one stage in-flight)");
    static_assert(EPI_PIPE_DEPTH >= 1, "EPI_PIPE_DEPTH must be >= 1");
    static_assert(CLC_PIPE_DEPTH >= 1, "CLC_PIPE_DEPTH must be >= 1");
    static_assert(SMEM_BYTES <= SMEM_BUDGET,
        "Smem footprint exceeds 227 KiB budget — reduce BM/BN/BK or LOAD_PIPE_DEPTH");
};

// Convenience aliases matching the validated configs used today
using Sm103Config_128x128x64 = Sm103ClusterConfig<128, 128, 64>;
using Sm103Config_256x256x64 = Sm103ClusterConfig<128, 256, 64>;

// ─── device primitives (hardware-level, identical for all layouts/configs) ───

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
        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
        :: "r"(addr), "r"(tx) : "memory");
}

__device__ __forceinline__
void mbarrier_arrive_expect_cluster(int addr, int tx)
{
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cluster.shared::cluster.b64 _, [%0], %1;"
        :: "r"(addr), "r"(tx) : "memory");
}

__device__ __forceinline__
void mbarrier_arrive_cluster(int addr)
{
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
        :: "r"(addr) : "memory");
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

__device__ __forceinline__
void cluster_fence_mbarrier_init() { asm volatile("fence.mbarrier_init.release.cluster;"); }

__device__ __forceinline__
void cluster_sync()
{
    asm volatile(
        "barrier.cluster.arrive.release.aligned;\n"
        "barrier.cluster.wait.acquire.aligned;\n"
        ::: "memory");
}

__device__ __forceinline__
uint32_t get_cluster_cta_rank()
{
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(r));
    return r;
}

__device__ __forceinline__
uint32_t map_smem_addr_to_cta_rank(uint32_t addr, uint32_t rank)
{
    uint32_t r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(r) : "r"(addr), "r"(rank));
    return r;
}

// 4D TMA: bulk async load [x × y × z × w] tile into shared memory
template<int CTA_GROUP = 1>
__device__ __forceinline__
void tma_4d_gmem2smem(int dst, const void* tmap,
                      int x, int y, int z, int w, int mbar)
{
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::%7"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap),
           "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar), "n"(CTA_GROUP)
        : "memory");
}

__device__ __forceinline__
constexpr uint64_t desc_encode(uint64_t x) { return (x & 0x3'FFFFULL) >> 4ULL; }

__device__ __forceinline__
uint64_t make_smem_desc(int addr)
{
    // SWIZZLE_W = 64 elements = 128 bytes — fixed for all SM103 configs
    constexpr int stride = 64 * 2 * 8; // SWIZZLE_W * sizeof(bf16) * 8 (swizzle factor)
    return desc_encode(addr)
         | (desc_encode(stride) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
}

template<int CTA_GROUP = 1>
__device__ __forceinline__
void tcgen05_mma_bf16(int tmem_addr,
                      uint64_t a_desc, uint64_t b_desc,
                      uint32_t i_desc, int use_accum)
{
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::%5.kind::f16 [%0], %1, %2, %3, p;\n\t"
        "}"
        :: "r"(tmem_addr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(use_accum), "n"(CTA_GROUP));
}

template<int CTA_GROUP = 1>
__device__ __forceinline__
void alloc_tmem(int tmem_smem_addr, int width)
{
    asm volatile("tcgen05.alloc.cta_group::%2.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(tmem_smem_addr), "r"(width), "n"(CTA_GROUP));
}

template<int CTA_GROUP = 1>
__device__ __forceinline__
void dealloc_tmem(int tmem_addr, int width)
{
    asm volatile("tcgen05.dealloc.cta_group::%2.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_addr), "r"(width), "n"(CTA_GROUP));
}

template<int CTA_GROUP = 2>
__device__ __forceinline__
void tcgen05_commit_multicast(int mbar_addr, uint16_t mask)
{
    asm volatile(
        "tcgen05.commit.cta_group::%2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(mbar_addr), "h"(mask), "n"(CTA_GROUP)
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

__device__ __forceinline__ void tcgen05_wait_ld()
    { asm volatile("tcgen05.wait::ld.sync.aligned;"); }
__device__ __forceinline__ void tcgen05_fence_before()
    { asm volatile("tcgen05.fence::before_thread_sync;"); }
__device__ __forceinline__ void tcgen05_fence_after()
    { asm volatile("tcgen05.fence::after_thread_sync;"); }
__device__ __forceinline__ void tma_store_fence()
    { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void tma_store_commit()
    { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }

template<int N>
__device__ __forceinline__
void tma_store_wait() { asm volatile("cp.async.bulk.wait_group %0;" :: "n"(N) : "memory"); }

__device__ __forceinline__
void tma_2d_smem2gmem(int src, const void* tmap, int x, int y)
{
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
        " [%0, {%2, %3}], [%1];"
        :: "l"(tmap), "r"(src), "r"(x), "r"(y) : "memory");
}

__device__ __forceinline__
void barrier_sync(int bar, int n)
{
    asm volatile("bar.sync %0, %1;" :: "r"(bar), "r"(n));
}

// ─── CLC primitives ──────────────────────────────────────────────────────────

__device__ __forceinline__
void clc_try_cancel(int response_addr, int mbar_addr)
{
    asm volatile(
        "clusterlaunchcontrol.try_cancel.async.shared::cta"
        ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];"
        :: "r"(response_addr), "r"(mbar_addr) : "memory");
}

__device__ __forceinline__
void clc_query_response(int response_addr, uint32_t& is_valid, uint32_t& new_ctaid)
{
    asm volatile(
        "{\n\t"
        ".reg .b128 handle;\n\t"
        ".reg .pred p;\n\t"
        "mov.s32 %0, 0;\n\t"
        "ld.shared.b128 handle, [%2];\n\t"
        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, handle;\n\t"
        "@p mov.s32 %0, 1;\n\t"
        "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %1, handle;\n\t"
        "}"
        : "=r"(is_valid), "=r"(new_ctaid)
        : "r"(response_addr) : "memory");
}

// ─── host-side TMA descriptor builders ───────────────────────────────────────

static inline void check_cu(CUresult e)
{
    if (e != CUDA_SUCCESS) {
        const char* msg = nullptr;
        cuGetErrorString(e, &msg);
        fprintf(stderr, "[BluBridge SM103] cuTensorMapEncodeTiled: %s (%d)\n",
                msg ? msg : "unknown", (int)e);
        exit(EXIT_FAILURE);
    }
}

// A[batch, M, K] row-major — NN and NT layouts (A is not transposed)
// TMA coords: (0, A_row, k_chunk, batch_idx)
static inline void init_4d_tma_A_MK(
    CUtensorMap* tmap, const __nv_bfloat16* ptr,
    int M, int K, int BM, int BK,
    uint64_t batchCount, uint64_t batchStride,
    CUtensorMapSwizzle swizzle)
{
    constexpr uint32_t rank = 4;
    uint64_t globalDim[rank]       = {64ULL, (uint64_t)M, (uint64_t)(K / 64), batchCount};
    uint64_t globalStrides[rank-1] = {(uint64_t)K * sizeof(__nv_bfloat16), 128ULL,
                                       batchStride * sizeof(__nv_bfloat16)};
    uint32_t boxDim[rank]          = {64U, (uint32_t)BM, (uint32_t)(BK / 64), 1U};
    uint32_t elementStrides[rank]  = {1U, 1U, 1U, 1U};
    check_cu(cuTensorMapEncodeTiled(tmap,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, (void*)ptr,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

// A[batch, K, M] row-major — TN layout (opA=T: A stored transposed)
// TMA coords: (0, k*BK, A_col, batch_idx)  where A_col = block_row*BM/SWIZZLE_W
static inline void init_4d_tma_A_KM(
    CUtensorMap* tmap, const __nv_bfloat16* ptr,
    int M, int K, int BM, int BK,
    uint64_t batchCount, uint64_t batchStride,
    CUtensorMapSwizzle swizzle)
{
    constexpr uint32_t rank = 4;
    uint64_t globalDim[rank]       = {64ULL, (uint64_t)K, (uint64_t)(M / 64), batchCount};
    uint64_t globalStrides[rank-1] = {(uint64_t)M * sizeof(__nv_bfloat16), 128ULL,
                                       batchStride * sizeof(__nv_bfloat16)};
    uint32_t boxDim[rank]          = {64U, (uint32_t)BK, (uint32_t)(BM / 64), 1U};
    uint32_t elementStrides[rank]  = {1U, 1U, 1U, 1U};
    check_cu(cuTensorMapEncodeTiled(tmap,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, (void*)ptr,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

// B[batch, K, N] row-major — NN and TN layouts (opB=N: B not transposed)
// TMA coords: (0, k*BK, B_col, batch_idx)  where B_col = (block_col*BN + rank*BN/2)/SWIZZLE_W
static inline void init_4d_tma_B_KN(
    CUtensorMap* tmap, const __nv_bfloat16* ptr,
    int K, int N, int BK, int BN,
    uint64_t batchCount, uint64_t batchStride,
    CUtensorMapSwizzle swizzle)
{
    constexpr uint32_t rank = 4;
    uint64_t globalDim[rank]       = {64ULL, (uint64_t)K, (uint64_t)(N / 64), batchCount};
    uint64_t globalStrides[rank-1] = {(uint64_t)N * sizeof(__nv_bfloat16), 128ULL,
                                       batchStride * sizeof(__nv_bfloat16)};
    uint32_t boxDim[rank]          = {64U, (uint32_t)BK,
                                      (uint32_t)(BN / 2 / 64), 1U};
    uint32_t elementStrides[rank]  = {1U, 1U, 1U, 1U};
    check_cu(cuTensorMapEncodeTiled(tmap,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, (void*)ptr,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

// B[batch, N, K] row-major — NT layout (opB=T: B stored transposed [N,K])
// TMA coords: (0, B_row, k_chunk, batch_idx)  where B_row = block_col*BN + rank*BN/2
static inline void init_4d_tma_B_NK(
    CUtensorMap* tmap, const __nv_bfloat16* ptr,
    int N, int K, int BN, int BK,
    uint64_t batchCount, uint64_t batchStride,
    CUtensorMapSwizzle swizzle)
{
    constexpr uint32_t rank = 4;
    uint64_t globalDim[rank]       = {64ULL, (uint64_t)N, (uint64_t)(K / 64), batchCount};
    uint64_t globalStrides[rank-1] = {(uint64_t)K * sizeof(__nv_bfloat16), 128ULL,
                                       batchStride * sizeof(__nv_bfloat16)};
    uint32_t boxDim[rank]          = {64U, (uint32_t)(BN / 2),
                                      (uint32_t)(BK / 64), 1U};
    uint32_t elementStrides[rank]  = {1U, 1U, 1U, 1U};
    check_cu(cuTensorMapEncodeTiled(tmap,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, (void*)ptr,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

// C[batch*M, N] flattened — identical for all layouts
// TMA coords: (c_col, c_row)
static inline void init_2d_tma_C(
    CUtensorMap* tmap, const __nv_bfloat16* ptr,
    uint64_t totalRows, uint64_t totalCols,
    uint32_t boxRows, uint32_t boxCols,
    CUtensorMapSwizzle swizzle)
{
    constexpr uint32_t rank = 2;
    uint64_t globalDim[rank]       = {totalCols, totalRows};
    uint64_t globalStrides[rank-1] = {totalCols * sizeof(__nv_bfloat16)};
    uint32_t boxDim[rank]          = {boxCols, boxRows};
    uint32_t elementStrides[rank]  = {1U, 1U};
    check_cu(cuTensorMapEncodeTiled(tmap,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, (void*)ptr,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

// ─── i_desc builder ───────────────────────────────────────────────────────────
//
// tcgen05 MMA instruction descriptor (compile-time constant per Config+Layout):
//   bits [4,7,10] : K-dimension sub-descriptors (always set)
//   bit  15       : trans_a = 1  → TN only   (A stored [K,M], read as M-major columns)
//   bit  16       : trans_b = 1  → NN and TN (B stored [K,N], read as row-major)
//                                → clear for NT (B stored [N,K])
//   bits [17..23] : BN >> 3      (N-tile width in 8-element units)
//   bits [24..30] : (CLUSTER_SIZE * BM) >> 4  (cluster M height in 16-element units)

template<typename Config, Sm103Layout Layout>
__host__ __device__ constexpr uint32_t sm103_make_i_desc()
{
    uint32_t d = (1U <<  4U)
               | (1U <<  7U)
               | (1U << 10U)
               | ((uint32_t)(Config::BN)                        >> 3U << 17U)
               | ((uint32_t)(Config::CLUSTER_SIZE * Config::BM) >> 4U << 24U);
    if constexpr (Layout == Sm103Layout::NN) d |= (1U << 16U);
    if constexpr (Layout == Sm103Layout::TN) d |= (1U << 15U) | (1U << 16U);
    // NT: no trans bits — B stored [N,K], MMA sees it as already column-major
    return d;
}

// ─── unified kernel ───────────────────────────────────────────────────────────
//
// Grid:  dim3(total_mn_tiles * batchCount * Config::CLUSTER_SIZE)
// Block: dim3(Config::NUM_THREADS)
// Smem:  Config::SMEM_BYTES dynamic shared memory

template<typename Config, Sm103Layout Layout>
__global__ __cluster_dims__(2, 1, 1)
__launch_bounds__(Config::NUM_THREADS, 1)
void bgemm_sm103_cluster_kernel(
    const __nv_bfloat16* __restrict__ A,    // global A (layout-dependent storage order)
    const __nv_bfloat16* __restrict__ C_in, // [batch, M, N] for beta read-back
          __nv_bfloat16* __restrict__ C,    // [batch, M, N] output
    int M, int N, int K,
    int total_mn_tiles,                     // (M / (CLUSTER_SIZE * BM)) * (N / BN)
    float alpha, float beta,
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B_tmap,
    const __grid_constant__ CUtensorMap C_tmap,
    int grid_n)                             // N / BN  (for on-device tile decode)
{
    // Unpack Config members into local aliases for readability
    constexpr int BM               = Config::BM;
    constexpr int BN               = Config::BN;
    constexpr int BK               = Config::BK;
    constexpr int LOAD_PIPE_DEPTH  = Config::LOAD_PIPE_DEPTH;
    constexpr int EPI_PIPE_DEPTH   = Config::EPI_PIPE_DEPTH;
    constexpr int CLC_PIPE_DEPTH   = Config::CLC_PIPE_DEPTH;
    constexpr int WARP_THREADS     = Config::WARP_THREADS;
    constexpr int SWIZZLE_W        = Config::SWIZZLE_W;
    constexpr int MMA_K_STEP       = Config::MMA_K_STEP;
    constexpr int CLUSTER_SIZE     = Config::CLUSTER_SIZE;
    constexpr int STORE_COLS       = Config::STORE_COLS;
    constexpr int STORE_PIPE_DEPTH = Config::STORE_PIPE_DEPTH;
    constexpr uint16_t CTA_MASK    = Config::CTA_MASK;

    constexpr int NUM_EPI_WARPS    = Config::NUM_EPI_WARPS;
    constexpr int PRODUCER_WARP    = Config::PRODUCER_WARP;
    constexpr int CONSUMER_WARP    = Config::CONSUMER_WARP;
    constexpr int SCHEDULER_WARP   = Config::SCHEDULER_WARP;
    constexpr int CLC_EMPTY_ARRIVE = Config::CLC_EMPTY_ARRIVE;

    const int      tid      = threadIdx.x;
    const int      warp_id  = tid / WARP_THREADS;
    const int      lane_id  = tid % WARP_THREADS;
    const uint32_t cta_rank = get_cluster_cta_rank();

    // ── shared memory layout ──────────────────────────────────────────────────
    extern __shared__ __align__(1024) char smem[];
    const int smem_ptr = static_cast<int>(__cvta_generic_to_shared(smem));

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ int tmem[1];
    const int tmem_smem_addr = static_cast<int>(__cvta_generic_to_shared(tmem));

    __shared__ __align__(8)  uint64_t full_mbar[LOAD_PIPE_DEPTH];
    __shared__ __align__(8)  uint64_t empty_mbar[LOAD_PIPE_DEPTH];
    __shared__ __align__(8)  uint64_t tmem_full[EPI_PIPE_DEPTH];
    __shared__ __align__(8)  uint64_t tmem_empty[EPI_PIPE_DEPTH];
    __shared__ __align__(16) uint8_t  clc_response[CLC_PIPE_DEPTH][16];
    __shared__ __align__(8)  uint64_t clc_full_mbar[CLC_PIPE_DEPTH];
    __shared__ __align__(8)  uint64_t clc_empty_mbar[CLC_PIPE_DEPTH];

    const int full_mbar_addr      = static_cast<int>(__cvta_generic_to_shared(full_mbar));
    const int empty_mbar_addr     = static_cast<int>(__cvta_generic_to_shared(empty_mbar));
    const int tmem_full_addr      = static_cast<int>(__cvta_generic_to_shared(tmem_full));
    const int tmem_empty_addr     = static_cast<int>(__cvta_generic_to_shared(tmem_empty));
    const int clc_response_addr   = static_cast<int>(__cvta_generic_to_shared(clc_response));
    const int clc_full_mbar_addr  = static_cast<int>(__cvta_generic_to_shared(clc_full_mbar));
    const int clc_empty_mbar_addr = static_cast<int>(__cvta_generic_to_shared(clc_empty_mbar));

    // Pipeline stage smem: A tile + B tile per stage
    // A smem: BM × BK bf16  (TN: [BK×BM] but same byte count — BK*BM == BM*BK)
    // B smem: (BN/2) × BK bf16  (each CTA holds half the N-column tile)
    constexpr int a_tile_bytes = Config::A_STAGE_BYTES;
    constexpr int b_tile_bytes = Config::B_STAGE_BYTES;
    constexpr int copy_size    = Config::STAGE_BYTES;
    constexpr int store_buf_sz = BM * STORE_COLS * 2;  // one store-smem slot (bytes)
    const int store_smem       = smem_ptr + LOAD_PIPE_DEPTH * copy_size;
    __nv_bfloat16* store_base  = reinterpret_cast<__nv_bfloat16*>(
                                   smem + LOAD_PIPE_DEPTH * copy_size);

    // ── initialisation ────────────────────────────────────────────────────────
    if (warp_id == PRODUCER_WARP && elect_sync())
    {
        for (int i = 0; i < LOAD_PIPE_DEPTH; ++i) {
            mbarrier_init(full_mbar_addr  + i * 8, 1 * CLUSTER_SIZE); // 1 producer × 2 CTAs
            mbarrier_init(empty_mbar_addr + i * 8, 1);
        }
        for (int i = 0; i < EPI_PIPE_DEPTH; ++i) {
            mbarrier_init(tmem_full_addr  + i * 8, 1);
            mbarrier_init(tmem_empty_addr + i * 8, NUM_EPI_WARPS * CLUSTER_SIZE);
        }
        for (int i = 0; i < CLC_PIPE_DEPTH; ++i) {
            mbarrier_init(clc_full_mbar_addr  + i * 8, 1);
            mbarrier_init(clc_empty_mbar_addr + i * 8, CLC_EMPTY_ARRIVE);
        }
        cluster_fence_mbarrier_init();
    }
    else if (warp_id == CONSUMER_WARP)
    {
        alloc_tmem<CLUSTER_SIZE>(tmem_smem_addr, BN * EPI_PIPE_DEPTH);
    }

    cluster_sync();

    const int tmem_addr = tmem[0];
    const int k_iters   = K / BK;

    // i_desc: compile-time constant encoding MMA shape + transposition for this layout
    constexpr uint32_t i_desc = sm103_make_i_desc<Config, Layout>();

    // CTA0 owns the real full_mbar; CTA1 reaches across the cluster to CTA0's copy
    const int tma_mbar_base = (cta_rank == 0)
        ? full_mbar_addr
        : map_smem_addr_to_cta_rank(full_mbar_addr, 0);

    const int cta0_clc_empty_addr = (cta_rank == 0)
        ? clc_empty_mbar_addr
        : map_smem_addr_to_cta_rank(clc_empty_mbar_addr, 0);

    const int init_tile_idx = (int)(blockIdx.x / CLUSTER_SIZE);
    int batch   = init_tile_idx / total_mn_tiles;
    int mn_tile = init_tile_idx % total_mn_tiles;

    // ── producer warp ─────────────────────────────────────────────────────────
    // Loads A and B tiles into the smem pipeline.
    // Layout-specific: how the global addresses map to TMA (x, y, z, w) coords.
    if (warp_id == PRODUCER_WARP && elect_sync())
    {
        int block_col = mn_tile % grid_n;
        int block_row = (mn_tile / grid_n) * CLUSTER_SIZE + (int)cta_rank;
        int cur_batch = batch;

        int stage = 0, phase = 0, issued = 0;
        int clc_stage = 0, clc_full_phase = 0;

        while (true)
        {
            if constexpr (Layout == Sm103Layout::TN)
            {
                // A[K,M]: select BM M-columns by column-group index
                // B[K,N]: select BN/2 N-columns by column-group index
                const int A_col = block_row * BM / SWIZZLE_W;
                const int B_col = (block_col * BN + (int)cta_rank * (BN / CLUSTER_SIZE))
                                  / SWIZZLE_W;

                for (int k = 0; k < k_iters; ++k)
                {
                    if (issued >= LOAD_PIPE_DEPTH)
                        mbarrier_wait(empty_mbar_addr + stage * 8, phase ^ 1);
                    ++issued;

                    const int A_smem = smem_ptr + stage * copy_size;
                    const int B_smem = A_smem + a_tile_bytes;
                    const int mbar   = tma_mbar_base + stage * 8;
                    mbarrier_arrive_expect(mbar, copy_size);

                    // A TN: y = k*BK (K-row), z = A_col (M col-group)
                    tma_4d_gmem2smem<CLUSTER_SIZE>(
                        A_smem, &A_tmap, 0, k * BK, A_col, cur_batch, mbar);
                    // B TN: y = k*BK (K-row), z = B_col (N col-group)
                    tma_4d_gmem2smem<CLUSTER_SIZE>(
                        B_smem, &B_tmap, 0, k * BK, B_col, cur_batch, mbar);

                    stage = (stage + 1) % LOAD_PIPE_DEPTH;
                    phase ^= (stage == 0);
                }
            }
            else if constexpr (Layout == Sm103Layout::NT)
            {
                // A[M,K]: select BM M-rows
                // B[N,K]: select BN/2 N-rows (B transposed — stored [N,K])
                const int A_row = block_row * BM;
                const int B_row = block_col * BN + (int)cta_rank * (BN / CLUSTER_SIZE);

                for (int k = 0; k < k_iters; ++k)
                {
                    if (issued >= LOAD_PIPE_DEPTH)
                        mbarrier_wait(empty_mbar_addr + stage * 8, phase ^ 1);
                    ++issued;

                    const int A_smem = smem_ptr + stage * copy_size;
                    const int B_smem = A_smem + a_tile_bytes;
                    const int mbar   = tma_mbar_base + stage * 8;
                    mbarrier_arrive_expect(mbar, copy_size);

                    // A NT: y = A_row (M-row), z = k (K-chunk index)
                    tma_4d_gmem2smem<CLUSTER_SIZE>(
                        A_smem, &A_tmap, 0, A_row, k, cur_batch, mbar);
                    // B NT: y = B_row (N-row), z = k (K-chunk index)
                    tma_4d_gmem2smem<CLUSTER_SIZE>(
                        B_smem, &B_tmap, 0, B_row, k, cur_batch, mbar);

                    stage = (stage + 1) % LOAD_PIPE_DEPTH;
                    phase ^= (stage == 0);
                }
            }
            else // NN
            {
                // A[M,K]: select BM M-rows
                // B[K,N]: select BN/2 N-columns by column-group index
                const int A_row = block_row * BM;
                const int B_col = (block_col * BN + (int)cta_rank * (BN / CLUSTER_SIZE))
                                  / SWIZZLE_W;

                for (int k = 0; k < k_iters; ++k)
                {
                    if (issued >= LOAD_PIPE_DEPTH)
                        mbarrier_wait(empty_mbar_addr + stage * 8, phase ^ 1);
                    ++issued;

                    const int A_smem = smem_ptr + stage * copy_size;
                    const int B_smem = A_smem + a_tile_bytes;
                    const int mbar   = tma_mbar_base + stage * 8;
                    mbarrier_arrive_expect(mbar, copy_size);

                    // A NN: y = A_row (M-row), z = k (K-chunk index)
                    tma_4d_gmem2smem<CLUSTER_SIZE>(
                        A_smem, &A_tmap, 0, A_row, k, cur_batch, mbar);
                    // B NN: y = k*BK (K-row), z = B_col (N col-group)
                    tma_4d_gmem2smem<CLUSTER_SIZE>(
                        B_smem, &B_tmap, 0, k * BK, B_col, cur_batch, mbar);

                    stage = (stage + 1) % LOAD_PIPE_DEPTH;
                    phase ^= (stage == 0);
                }
            }

            mbarrier_wait(clc_full_mbar_addr + clc_stage * 8, clc_full_phase);
            uint32_t is_valid, new_ctaid;
            clc_query_response(clc_response_addr + clc_stage * 16, is_valid, new_ctaid);
            mbarrier_arrive_cluster(cta0_clc_empty_addr + clc_stage * 8);
            clc_stage = (clc_stage + 1) % CLC_PIPE_DEPTH;
            if (clc_stage == 0) clc_full_phase ^= 1;

            if (!is_valid) break;

            const int new_tile_idx = (int)(new_ctaid / CLUSTER_SIZE);
            cur_batch = new_tile_idx / total_mn_tiles;
            const int new_mn = new_tile_idx % total_mn_tiles;
            block_col = new_mn % grid_n;
            block_row = (new_mn / grid_n) * CLUSTER_SIZE + (int)cta_rank;
            issued    = 0;
        }
    }
    // ── consumer warp ─────────────────────────────────────────────────────────
    // Drives tcgen05.mma from CTA0.  Layout-specific: a_off / b_off reflect
    // how the K dimension strides through smem for each storage order.
    else if (warp_id == CONSUMER_WARP && elect_sync())
    {
        int stage = 0, phase = 0;
        int wave_iter = 0;
        int clc_stage = 0, clc_full_phase = 0;

        while (true)
        {
            if (cta_rank == 0)
            {
                const int wave_stage = wave_iter % EPI_PIPE_DEPTH;
                const int wave_phase = (wave_iter / EPI_PIPE_DEPTH) & 1;

                if (wave_iter >= EPI_PIPE_DEPTH)
                    mbarrier_wait(tmem_empty_addr + wave_stage * 8, wave_phase ^ 1);

                tcgen05_fence_after();

                for (int k = 0; k < k_iters; ++k)
                {
                    mbarrier_wait(full_mbar_addr + stage * 8, phase);
                    tcgen05_fence_after();

                    const int A_smem = smem_ptr + stage * copy_size;
                    const int B_smem = A_smem + a_tile_bytes;

                    for (int k1 = 0; k1 < BK / SWIZZLE_W; ++k1)
                    {
                        for (int k2 = 0; k2 < SWIZZLE_W / MMA_K_STEP; ++k2)
                        {
                            const int use_accum = (k == 0 && k1 == 0 && k2 == 0) ? 0 : 1;

                            // a_off: byte offset into A smem for K-step (k1, k2)
                            // NN/NT: A smem is [BM × BK], K advances along columns
                            //        → k2 * MMA_K_STEP columns = k2 * MMA_K_STEP * 2 bytes
                            // TN:    A smem is [BK × BM], K advances along rows
                            //        → k2 * MMA_K_STEP rows = k2 * MMA_K_STEP * BM * 2 bytes
                            int a_off, b_off;
                            if constexpr (Layout == Sm103Layout::TN) {
                                a_off = k1 * SWIZZLE_W * BM * 2
                                      + k2 * MMA_K_STEP  * BM * 2;
                            } else {
                                a_off = k1 * SWIZZLE_W * BM * 2
                                      + k2 * MMA_K_STEP  * 2;
                            }

                            // b_off: byte offset into B smem for K-step (k1, k2)
                            // NN/TN: B smem is [BK × BN/2], K advances along rows
                            //        → k2 * MMA_K_STEP rows = k2 * MMA_K_STEP * 64 * 2 bytes
                            // NT:    B smem is [BN/2 × BK], K advances along columns
                            //        → k2 * MMA_K_STEP columns = k2 * MMA_K_STEP * 2 bytes
                            if constexpr (Layout == Sm103Layout::NT) {
                                b_off = k1 * SWIZZLE_W * (BN / CLUSTER_SIZE) * 2
                                      + k2 * MMA_K_STEP  * 2;
                            } else {
                                b_off = k1 * SWIZZLE_W * SWIZZLE_W * 2
                                      + k2 * MMA_K_STEP  * SWIZZLE_W * 2;
                            }

                            tcgen05_mma_bf16<CLUSTER_SIZE>(
                                tmem_addr + wave_stage * BN,
                                make_smem_desc(A_smem + a_off),
                                make_smem_desc(B_smem + b_off),
                                i_desc, use_accum);
                        }
                    }

                    tcgen05_commit_multicast<CLUSTER_SIZE>(
                        empty_mbar_addr + stage * 8, CTA_MASK);

                    stage = (stage + 1) % LOAD_PIPE_DEPTH;
                    phase ^= (stage == 0);
                }

                tcgen05_commit_multicast<CLUSTER_SIZE>(
                    tmem_full_addr + wave_stage * 8, CTA_MASK);
                wave_iter++;
            }

            mbarrier_wait(clc_full_mbar_addr + clc_stage * 8, clc_full_phase);
            uint32_t is_valid, new_ctaid;
            clc_query_response(clc_response_addr + clc_stage * 16, is_valid, new_ctaid);
            mbarrier_arrive_cluster(cta0_clc_empty_addr + clc_stage * 8);
            clc_stage = (clc_stage + 1) % CLC_PIPE_DEPTH;
            if (clc_stage == 0) clc_full_phase ^= 1;

            if (!is_valid) break;
        }
    }
    // ── scheduler warp (identical for all layouts and configs) ───────────────
    else if (warp_id == SCHEDULER_WARP && elect_sync())
    {
        if (cta_rank == 0)
        {
            #pragma unroll
            for (int s = 0; s < CLC_PIPE_DEPTH; ++s)
            {
                mbarrier_arrive_expect_cluster(clc_full_mbar_addr + s * 8, 16);
                int remote_full = map_smem_addr_to_cta_rank(
                    clc_full_mbar_addr + s * 8, 1);
                mbarrier_arrive_expect_cluster(remote_full, 16);
                clc_try_cancel(clc_response_addr + s * 16, clc_full_mbar_addr + s * 8);
            }
        }

        int clc_stage = 0, clc_full_phase = 0, clc_empty_phase = 0;

        while (true)
        {
            mbarrier_wait(clc_full_mbar_addr + clc_stage * 8, clc_full_phase);
            uint32_t is_valid, new_ctaid;
            clc_query_response(clc_response_addr + clc_stage * 16, is_valid, new_ctaid);
            mbarrier_arrive_cluster(cta0_clc_empty_addr + clc_stage * 8);

            int issue_stage = clc_stage;
            clc_stage = (clc_stage + 1) % CLC_PIPE_DEPTH;
            if (clc_stage == 0) clc_full_phase ^= 1;

            if (!is_valid) break;

            if (cta_rank == 0)
            {
                mbarrier_wait(clc_empty_mbar_addr + issue_stage * 8, clc_empty_phase);
                if (issue_stage == CLC_PIPE_DEPTH - 1) clc_empty_phase ^= 1;

                mbarrier_arrive_expect_cluster(clc_full_mbar_addr + issue_stage * 8, 16);
                int remote_full = map_smem_addr_to_cta_rank(
                    clc_full_mbar_addr + issue_stage * 8, 1);
                mbarrier_arrive_expect_cluster(remote_full, 16);
                clc_try_cancel(clc_response_addr + issue_stage * 16,
                               clc_full_mbar_addr + issue_stage * 8);
            }
        }
    }
    // ── epilogue warps (identical for all layouts and configs) ───────────────
    // Reads TMEM → scales by alpha/beta → writes to smem → TMA stores to C.
    else if (warp_id < NUM_EPI_WARPS)
    {
        constexpr int EPI_BAR     = 7;
        constexpr int EPI_THREADS = NUM_EPI_WARPS * WARP_THREADS;
        constexpr int num_chunks  = BN / STORE_COLS;
        constexpr int LOADS_PER   = STORE_COLS / 8;  // 8 = tcgen05_ld loads 8 floats per call

        int block_col = mn_tile % grid_n;
        int block_row = (mn_tile / grid_n) * CLUSTER_SIZE + (int)cta_rank;
        int cur_batch = batch;

        int wave_iter = 0;
        int clc_stage = 0, clc_full_phase = 0;

        while (true)
        {
            const int wave_stage = wave_iter % EPI_PIPE_DEPTH;
            const int wave_phase = (wave_iter / EPI_PIPE_DEPTH) & 1;

            mbarrier_wait(tmem_full_addr + wave_stage * 8, wave_phase);
            tcgen05_fence_after();

            // CTA0 → TMEM rows [0 .. BM), CTA1 → rows [BM .. 2*BM)
            const int tmem_row = (tmem_addr + wave_stage * BN)
                               + ((int(cta_rank) * BM + warp_id * WARP_THREADS) << 16);
            const int row = warp_id * WARP_THREADS + lane_id;
            int store_stage = 0;

            for (int chunk = 0; chunk < num_chunks; ++chunk)
            {
                if (warp_id == 0) tma_store_wait<STORE_PIPE_DEPTH - 1>();

                float tmp[LOADS_PER][8];
                #pragma unroll
                for (int n = 0; n < LOADS_PER; ++n)
                    tcgen05_ld(tmp[n], tmem_row + chunk * STORE_COLS + n * 8);
                tcgen05_wait_ld();

                if (chunk == num_chunks - 1)
                {
                    tcgen05_fence_before();
                    // bit 24 selects CTA; clear it to always address CTA0's tmem_empty
                    const int tmem_empty_cta0 = (tmem_empty_addr + wave_stage * 8) & 0xFEFFFFFF;
                    if (elect_sync())
                        mbarrier_arrive_cluster(tmem_empty_cta0);
                }

                barrier_sync(EPI_BAR, EPI_THREADS);

                #pragma unroll
                for (int n = 0; n < LOADS_PER; ++n)
                {
                    __nv_bfloat162 packed[4];

                    if (beta != 0.0f)
                    {
                        const int c_col = block_col * BN + chunk * STORE_COLS + n * 8;
                        const int c_row = cur_batch * M + block_row * BM + row;
                        const __nv_bfloat16* cp = C_in + (long long)c_row * N + c_col;
                        #pragma unroll
                        for (int i = 0; i < 4; ++i)
                        {
                            __nv_bfloat162 old =
                                *reinterpret_cast<const __nv_bfloat162*>(cp + i * 2);
                            float2 old_f = __bfloat1622float2(old);
                            float v0 = tmp[n][i * 2]     * alpha + old_f.x * beta;
                            float v1 = tmp[n][i * 2 + 1] * alpha + old_f.y * beta;
                            packed[i] = __float22bfloat162_rn({v0, v1});
                        }
                    }
                    else
                    {
                        #pragma unroll
                        for (int i = 0; i < 4; ++i)
                        {
                            packed[i] = __float22bfloat162_rn(
                                {tmp[n][i * 2] * alpha, tmp[n][i * 2 + 1] * alpha});
                        }
                    }

                    // Swizzle n within the store-smem row to avoid bank conflicts
                    const int swizzled_n = n ^ (row & 7);
                    __nv_bfloat16* wp = store_base
                                      + store_stage * BM * STORE_COLS
                                      + row * STORE_COLS + swizzled_n * 8;
                    *reinterpret_cast<int4*>(wp) = *reinterpret_cast<int4*>(packed);
                }

                __syncwarp();
                tma_store_fence();
                barrier_sync(EPI_BAR, EPI_THREADS);

                if (warp_id == 0 && elect_sync())
                {
                    const int src   = store_smem + store_stage * store_buf_sz;
                    const int c_col = block_col * BN + chunk * STORE_COLS;
                    const int c_row = cur_batch * M + block_row * BM;
                    tma_2d_smem2gmem(src, &C_tmap, c_col, c_row);
                    tma_store_commit();
                }

                store_stage ^= 1;
            }

            wave_iter++;

            mbarrier_wait(clc_full_mbar_addr + clc_stage * 8, clc_full_phase);
            uint32_t is_valid, new_ctaid;
            clc_query_response(clc_response_addr + clc_stage * 16, is_valid, new_ctaid);
            if (elect_sync())
                mbarrier_arrive_cluster(cta0_clc_empty_addr + clc_stage * 8);
            clc_stage = (clc_stage + 1) % CLC_PIPE_DEPTH;
            if (clc_stage == 0) clc_full_phase ^= 1;

            if (!is_valid) break;

            const int new_tile_idx = (int)(new_ctaid / CLUSTER_SIZE);
            cur_batch  = new_tile_idx / total_mn_tiles;
            const int new_mn = new_tile_idx % total_mn_tiles;
            block_col = new_mn % grid_n;
            block_row = (new_mn / grid_n) * CLUSTER_SIZE + (int)cta_rank;
        }

        if (warp_id == 0) tma_store_wait<0>();
        barrier_sync(EPI_BAR, EPI_THREADS);
    }

    cluster_sync();
    if (warp_id == 0)
        dealloc_tmem<CLUSTER_SIZE>(tmem_addr, BN * EPI_PIPE_DEPTH);
}

// ─── unified launcher ─────────────────────────────────────────────────────────
//
// Template args:
//   Config  — an Sm103ClusterConfig<BM, BN, BK, ...> instantiation
//   Layout  — Sm103Layout::NN / NT / TN
//
// Replaces the three separate launch_bgemm_{nn,nt,tn}_cluster functions.

template<typename Config, Sm103Layout Layout>
static void launch_bgemm_sm103_cluster(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
          __nv_bfloat16* C,
    int M, int N, int K, int batchCount,
    float alpha, float beta,
    cudaStream_t stream)
{
    static int s_arch_ok = -1;
    if (s_arch_ok < 0) {
        int dev; cudaGetDevice(&dev);
        cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
        s_arch_ok = (prop.major == 10 && prop.minor >= 3) ? 1 : 0;
        if (!s_arch_ok)
            fprintf(stderr,
                "[BluBridge SM103] WARNING: device '%s' SM %d.%d, requires SM 10.3+\n",
                prop.name, prop.major, prop.minor);
    }
    if (!s_arch_ok) return;

    constexpr int BM           = Config::BM;
    constexpr int BN           = Config::BN;
    constexpr int BK           = Config::BK;
    constexpr int CLUSTER_SIZE = Config::CLUSTER_SIZE;
    constexpr int STORE_COLS   = Config::STORE_COLS;

    // Build TMA descriptors — layout-specific A and B initialisation
    CUtensorMap A_tmap, B_tmap, C_tmap;

    if constexpr (Layout == Sm103Layout::TN) {
        // A[batch, K, M]  batch stride = K * M
        init_4d_tma_A_KM(&A_tmap, A, M, K, BM, BK,
                          (uint64_t)batchCount, (uint64_t)K * M,
                          CU_TENSOR_MAP_SWIZZLE_128B);
        // B[batch, K, N]  batch stride = K * N
        init_4d_tma_B_KN(&B_tmap, B, K, N, BK, BN,
                          (uint64_t)batchCount, (uint64_t)K * N,
                          CU_TENSOR_MAP_SWIZZLE_128B);
    } else if constexpr (Layout == Sm103Layout::NT) {
        // A[batch, M, K]  batch stride = M * K
        init_4d_tma_A_MK(&A_tmap, A, M, K, BM, BK,
                          (uint64_t)batchCount, (uint64_t)M * K,
                          CU_TENSOR_MAP_SWIZZLE_128B);
        // B[batch, N, K]  batch stride = N * K  (stored transposed)
        init_4d_tma_B_NK(&B_tmap, B, N, K, BN, BK,
                          (uint64_t)batchCount, (uint64_t)N * K,
                          CU_TENSOR_MAP_SWIZZLE_128B);
    } else { // NN
        // A[batch, M, K]  batch stride = M * K
        init_4d_tma_A_MK(&A_tmap, A, M, K, BM, BK,
                          (uint64_t)batchCount, (uint64_t)M * K,
                          CU_TENSOR_MAP_SWIZZLE_128B);
        // B[batch, K, N]  batch stride = K * N
        init_4d_tma_B_KN(&B_tmap, B, K, N, BK, BN,
                          (uint64_t)batchCount, (uint64_t)K * N,
                          CU_TENSOR_MAP_SWIZZLE_128B);
    }

    init_2d_tma_C(&C_tmap, C,
                  (uint64_t)batchCount * M, (uint64_t)N,
                  (uint32_t)BM, (uint32_t)STORE_COLS,
                  CU_TENSOR_MAP_SWIZZLE_128B);

    const int grid_m         = M / (CLUSTER_SIZE * BM);
    const int grid_n         = N / BN;
    const int total_mn_tiles = grid_m * grid_n;

    auto kernel = bgemm_sm103_cluster_kernel<Config, Layout>;
    cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, Config::SMEM_BYTES);

    dim3 grid(total_mn_tiles * batchCount * CLUSTER_SIZE);

    kernel<<<grid, Config::NUM_THREADS, Config::SMEM_BYTES, stream>>>(
        A, C, C,
        M, N, K, total_mn_tiles,
        alpha, beta,
        A_tmap, B_tmap, C_tmap,
        grid_n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "[BluBridge SM103 Cluster] Launch error: %s\n",
                cudaGetErrorString(err));
}
