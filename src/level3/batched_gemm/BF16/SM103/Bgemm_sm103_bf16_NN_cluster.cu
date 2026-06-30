// Bgemm_sm103_bf16_NN_cluster.cu
// Blackwell Ultra (SM103a / B300) BF16 Batched GEMM — NN layout, 2-CTA cluster
// C[b,M,N] = alpha * A[b,M,K] × B[b,K,N] + beta * C[b,M,N]
//
// Architecture (mirrors kernel_v8 / ThunderKittens bf16_b200_gemm):
//   - tcgen05.mma.cta_group::2.kind::f16  (2-CTA cooperative MMA)
//   - CLC (Cluster Launch Control) persistent tile scheduling
//   - Hilbert-curve tile ordering for L2 cache locality
//   - 7 warps: 0-3 epilogue | 4 producer | 5 consumer | 6 scheduler
//   - 4D TMA bulk async A/B loads (batch dim), 2D TMA store for C
//   - Double-buffered TMEM (NUM_EPI_STAGES = 2)
//   - trans_b = 1 in i_desc for NN layout (B stored [K,N])
//
// Each cluster covers:  CTA0 → rows [block_row0*BM .. (block_row0+1)*BM-1]
//                       CTA1 → rows [block_row1*BM .. (block_row1+1)*BM-1]
// (block_row1 = block_row0 + 1 via Hilbert map), columns [block_col*BN .. (block_col+1)*BN-1].
// CTA0 loads the low  BN/2 columns of B; CTA1 loads the high BN/2 columns.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "mycublas.h"

// ─── compile-time constants ───────────────────────────────────────────────────

static constexpr int WARP_SIZE        = 32;
static constexpr int SWIZZLE_W        = 64;   // 64 elements = 128 bytes per swizzle group
static constexpr int MMA_K_STEP       = 16;   // inner-K step of one tcgen05 MMA call
static constexpr int CTA_GROUP_SIZE   = 2;    // cluster width
static constexpr uint16_t CTA_MASK    = 0b11; // 2-CTA multicast mask
// NUM_EPI_WARPS = BM/WARP_SIZE — computed inside the kernel template (scales with BM)
static constexpr int NUM_PROD_WARPS   = 1;
static constexpr int NUM_CONS_WARPS   = 1;
static constexpr int NUM_SCHED_WARPS  = 1;
static constexpr int STORE_N          = 64;   // columns per TMA store call
static constexpr int NUM_STORE_STGS   = 2;    // double-buffer store smem
static constexpr int NUM_EPI_STAGES   = 2;    // TMEM double-buffering slots
static constexpr int NUM_CLC_STAGES   = 2;    // CLC pipeline depth

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
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(addr), "r"(count));
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
void cluster_fence_mbarrier_init()
{
    asm volatile("fence.mbarrier_init.release.cluster;");
}

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

// 4D TMA: loads [box0 × box1 × box2 × box3] tile at coords (x,y,z,w)
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
constexpr uint64_t desc_encode(uint64_t x)
{
    return (x & 0x3'FFFFULL) >> 4ULL;
}

__device__ __forceinline__
uint64_t make_smem_desc(int addr)
{
    const int stride = 8 * SWIZZLE_W * static_cast<int>(sizeof(__nv_bfloat16));
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
    asm volatile(
        "tcgen05.alloc.cta_group::%2.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(tmem_smem_addr), "r"(width), "n"(CTA_GROUP));
}

template<int CTA_GROUP = 1>
__device__ __forceinline__
void dealloc_tmem(int tmem_addr, int width)
{
    asm volatile(
        "tcgen05.dealloc.cta_group::%2.sync.aligned.b32 %0, %1;"
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

__device__ __forceinline__ void tcgen05_wait_ld()   { asm volatile("tcgen05.wait::ld.sync.aligned;"); }
__device__ __forceinline__ void tcgen05_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;"); }
__device__ __forceinline__ void tcgen05_fence_after()  { asm volatile("tcgen05.fence::after_thread_sync;"); }
__device__ __forceinline__ void tma_store_fence()   { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void tma_store_commit()  { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }

template<int N>
__device__ __forceinline__
void tma_store_wait() { asm volatile("cp.async.bulk.wait_group %0;" :: "n"(N) : "memory"); }

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
void clc_query_response(int response_addr, uint32_t &is_valid, uint32_t &new_ctaid)
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

// ─── host-side helpers ────────────────────────────────────────────────────────

static inline void check_cu(CUresult e)
{
    if (e != CUDA_SUCCESS) {
        const char* msg = nullptr;
        cuGetErrorString(e, &msg);
        fprintf(stderr, "[BluBridge TMA] cuTensorMapEncodeTiled: %s (%d)\n",
                msg ? msg : "unknown", (int)e);
        exit(EXIT_FAILURE);
    }
}

// 4D TMA for A: A[batch, M, K] row-major, box = [64, BM, BK/64, 1]
static inline void init_4d_tma_A(
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

// 4D TMA for B NN: B[batch, K, N] row-major, box = [64, BK, BN/(CTA_GROUP_SIZE*64), 1]
static inline void init_4d_tma_B_NN(
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
                                      (uint32_t)(BN / CTA_GROUP_SIZE / 64), 1U};
    uint32_t elementStrides[rank]  = {1U, 1U, 1U, 1U};
    check_cu(cuTensorMapEncodeTiled(tmap,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, (void*)ptr,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

// 2D TMA for C store: C[batch*M, N] flattened, box = [STORE_N, BM]
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

// ─── kernel ───────────────────────────────────────────────────────────────────
//
// Grid: dim3(total_mn_tiles * batchCount * CTA_GROUP_SIZE)
// Warp assignments:
//   0-3 : epilogue  — tcgen05.ld TMEM → convert+scale → TMA store
//   4   : producer  — TMA 4D loads A and B tiles into smem
//   5   : consumer  — tcgen05.mma (CTA 0 only drives MMA)
//   6   : scheduler — CLC try_cancel/query loop

template<int BM, int BN, int BK, int QUEUE_SIZE>
__global__ __cluster_dims__(2, 1, 1)
void bgemm_sm103_bf16_nn_cluster_kernel(
    const __nv_bfloat16* __restrict__ A,   // [batch, M, K] row-major
    const __nv_bfloat16* __restrict__ C_in,// [batch, M, N] row-major (for beta)
          __nv_bfloat16* __restrict__ C,   // [batch, M, N] row-major (output)
    int M, int N, int K,
    int total_mn_tiles,                    // (M/(CTA_GROUP_SIZE*BM)) * (N/BN)
    float alpha, float beta,
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B_tmap,
    const __grid_constant__ CUtensorMap C_tmap,
    int grid_n)                            // number of BN-wide column tiles (N/BN)
{
    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const uint32_t cta_rank = get_cluster_cta_rank();

    // ── shared memory layout ──────────────────────────────────────────────────
    // [QUEUE_SIZE × (BM*BK + (BN/2)*BK)] pipeline stages, then store double-buffer
    extern __shared__ __align__(1024) char smem[];
    const int smem_ptr = static_cast<int>(__cvta_generic_to_shared(smem));

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ int tmem[1];
    const int tmem_smem_addr = static_cast<int>(__cvta_generic_to_shared(tmem));

    __shared__ __align__(8) uint64_t full_mbar[QUEUE_SIZE];
    __shared__ __align__(8) uint64_t empty_mbar[QUEUE_SIZE];
    __shared__ __align__(8) uint64_t tmem_full[NUM_EPI_STAGES];
    __shared__ __align__(8) uint64_t tmem_empty[NUM_EPI_STAGES];
    __shared__ __align__(16) uint8_t  clc_response[NUM_CLC_STAGES][16];
    __shared__ __align__(8)  uint64_t clc_full_mbar[NUM_CLC_STAGES];
    __shared__ __align__(8)  uint64_t clc_empty_mbar[NUM_CLC_STAGES];

    const int full_mbar_addr       = static_cast<int>(__cvta_generic_to_shared(full_mbar));
    const int empty_mbar_addr      = static_cast<int>(__cvta_generic_to_shared(empty_mbar));
    const int tmem_full_addr       = static_cast<int>(__cvta_generic_to_shared(tmem_full));
    const int tmem_empty_addr      = static_cast<int>(__cvta_generic_to_shared(tmem_empty));
    const int clc_response_addr    = static_cast<int>(__cvta_generic_to_shared(clc_response));
    const int clc_full_mbar_addr   = static_cast<int>(__cvta_generic_to_shared(clc_full_mbar));
    const int clc_empty_mbar_addr  = static_cast<int>(__cvta_generic_to_shared(clc_empty_mbar));

    // Warp layout scales with BM (like ThunderKittens' variable warpgroup count):
    //   epilogue warps: 0 .. NUM_EPI_WARPS-1  (BM/32 warps, one per 32 M-rows)
    //   producer: NUM_EPI_WARPS, consumer: +1, scheduler: +2
    constexpr int NUM_EPI_WARPS    = BM / WARP_SIZE;
    constexpr int CLC_EMPTY_ARRIVE = (NUM_EPI_WARPS + NUM_PROD_WARPS + NUM_CONS_WARPS + NUM_SCHED_WARPS) * CTA_GROUP_SIZE;
    constexpr int producer_warp    = NUM_EPI_WARPS;
    constexpr int consumer_warp    = NUM_EPI_WARPS + 1;
    constexpr int scheduler_warp   = NUM_EPI_WARPS + 2;

    // Per-stage smem: each CTA holds full BM rows of A and BN/2 cols of B
    constexpr int a_tile_bytes = BM * BK * (int)sizeof(__nv_bfloat16);
    constexpr int b_tile_bytes = (BN / CTA_GROUP_SIZE) * BK * (int)sizeof(__nv_bfloat16);
    constexpr int copy_size    = a_tile_bytes + b_tile_bytes;
    constexpr int store_buf_sz = BM * STORE_N * (int)sizeof(__nv_bfloat16);
    const int store_smem       = smem_ptr + QUEUE_SIZE * copy_size;
    __nv_bfloat16* store_base  = reinterpret_cast<__nv_bfloat16*>(smem + QUEUE_SIZE * copy_size);

    // ── initialisation ────────────────────────────────────────────────────────
    if (warp_id == producer_warp && elect_sync())
    {
        for (int i = 0; i < QUEUE_SIZE; ++i) {
            mbarrier_init(full_mbar_addr  + i * 8, NUM_PROD_WARPS * CTA_GROUP_SIZE);
            mbarrier_init(empty_mbar_addr + i * 8, 1);
        }
        for (int i = 0; i < NUM_EPI_STAGES; ++i) {
            mbarrier_init(tmem_full_addr  + i * 8, 1);
            mbarrier_init(tmem_empty_addr + i * 8, NUM_EPI_WARPS * CTA_GROUP_SIZE);
        }
        for (int i = 0; i < NUM_CLC_STAGES; ++i) {
            mbarrier_init(clc_full_mbar_addr  + i * 8, 1);
            mbarrier_init(clc_empty_mbar_addr + i * 8, CLC_EMPTY_ARRIVE);
        }
        cluster_fence_mbarrier_init();
    }
    else if (warp_id == consumer_warp)
    {
        alloc_tmem<CTA_GROUP_SIZE>(tmem_smem_addr, BN * NUM_EPI_STAGES);
    }

    cluster_sync();

    const int tmem_addr = tmem[0];
    const int k_iters   = K / BK;

    // i_desc for NN (trans_b=1, M = CTA_GROUP_SIZE*BM, N = BN)
    constexpr uint32_t i_desc =
          (1U <<  4U)
        | (1U <<  7U)
        | (1U << 10U)
        | (1U << 16U)                                       // trans_b = 1 for NN
        | ((uint32_t)(BN) >> 3U << 17U)
        | ((uint32_t)(CTA_GROUP_SIZE * BM) >> 4U << 24U);

    // CTA0 owns full_mbar; CTA1 maps to CTA0's address
    const int tma_mbar_base = (cta_rank == 0)
        ? full_mbar_addr
        : map_smem_addr_to_cta_rank(full_mbar_addr, 0);

    // tmem_empty and clc_empty live on CTA0
    const int cta0_clc_empty_addr = (cta_rank == 0)
        ? clc_empty_mbar_addr
        : map_smem_addr_to_cta_rank(clc_empty_mbar_addr, 0);

    // Decode initial tile assignment
    const int init_tile_idx    = (int)(blockIdx.x / CTA_GROUP_SIZE);
    int batch     = init_tile_idx / total_mn_tiles;
    int mn_tile   = init_tile_idx % total_mn_tiles;

    // ── producer warp ─────────────────────────────────────────────────────────
    if (warp_id == producer_warp && elect_sync())
    {
        int block_col = mn_tile % grid_n;
        int block_row = (mn_tile / grid_n) * CTA_GROUP_SIZE + (int)cta_rank;
        int cur_batch = batch;

        int stage = 0, phase = 0, issued = 0;
        int clc_stage = 0, clc_full_phase = 0;

        while (true)
        {
            const int A_row = block_row * BM;
            const int B_col = (block_col * BN + (int)cta_rank * (BN / CTA_GROUP_SIZE)) / SWIZZLE_W;

            for (int k = 0; k < k_iters; ++k)
            {
                if (issued >= QUEUE_SIZE)
                    mbarrier_wait(empty_mbar_addr + stage * 8, phase ^ 1);
                ++issued;

                const int A_smem = smem_ptr + stage * copy_size;
                const int B_smem = A_smem + a_tile_bytes;
                const int mbar   = tma_mbar_base + stage * 8;

                mbarrier_arrive_expect(mbar, copy_size);

                tma_4d_gmem2smem<CTA_GROUP_SIZE>(
                    A_smem, &A_tmap,
                    0, A_row, k, cur_batch, mbar);
                tma_4d_gmem2smem<CTA_GROUP_SIZE>(
                    B_smem, &B_tmap,
                    0, k * BK, B_col, cur_batch, mbar);

                stage = (stage + 1) % QUEUE_SIZE;
                phase ^= (stage == 0);
            }

            mbarrier_wait(clc_full_mbar_addr + clc_stage * 8, clc_full_phase);
            uint32_t is_valid, new_ctaid;
            clc_query_response(clc_response_addr + clc_stage * 16, is_valid, new_ctaid);
            mbarrier_arrive_cluster(cta0_clc_empty_addr + clc_stage * 8);
            clc_stage = (clc_stage + 1) % NUM_CLC_STAGES;
            if (clc_stage == 0) clc_full_phase ^= 1;

            if (!is_valid) break;

            const int new_tile_idx = (int)(new_ctaid / CTA_GROUP_SIZE);
            cur_batch  = new_tile_idx / total_mn_tiles;
            const int new_mn = new_tile_idx % total_mn_tiles;
            block_col  = new_mn % grid_n;
            block_row  = (new_mn / grid_n) * CTA_GROUP_SIZE + (int)cta_rank;
            issued     = 0;
        }
    }
    // ── consumer warp ─────────────────────────────────────────────────────────
    else if (warp_id == consumer_warp && elect_sync())
    {
        int stage = 0, phase = 0;
        int wave_iter = 0;
        int clc_stage = 0, clc_full_phase = 0;

        while (true)
        {
            if (cta_rank == 0)
            {
                const int wave_stage = wave_iter % NUM_EPI_STAGES;
                const int wave_phase = (wave_iter / NUM_EPI_STAGES) & 1;

                if (wave_iter >= NUM_EPI_STAGES)
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

                            // a_off: step along K within [BM × BK] A smem
                            const int a_off =
                                k1 * SWIZZLE_W * BM * (int)sizeof(__nv_bfloat16)
                              + k2 * MMA_K_STEP  * (int)sizeof(__nv_bfloat16);

                            // b_off for NN: B smem is [BK × BN/2], step along K rows
                            const int b_off =
                                k1 * SWIZZLE_W * (BN / CTA_GROUP_SIZE) * (int)sizeof(__nv_bfloat16)
                              + k2 * MMA_K_STEP  * (BN / CTA_GROUP_SIZE) * (int)sizeof(__nv_bfloat16);

                            tcgen05_mma_bf16<CTA_GROUP_SIZE>(
                                tmem_addr + wave_stage * BN,
                                make_smem_desc(A_smem + a_off),
                                make_smem_desc(B_smem + b_off),
                                i_desc, use_accum);
                        }
                    }

                    tcgen05_commit_multicast<CTA_GROUP_SIZE>(
                        empty_mbar_addr + stage * 8, CTA_MASK);

                    stage = (stage + 1) % QUEUE_SIZE;
                    phase ^= (stage == 0);
                }

                tcgen05_commit_multicast<CTA_GROUP_SIZE>(
                    tmem_full_addr + wave_stage * 8, CTA_MASK);
                wave_iter++;
            }

            mbarrier_wait(clc_full_mbar_addr + clc_stage * 8, clc_full_phase);
            uint32_t is_valid, new_ctaid;
            clc_query_response(clc_response_addr + clc_stage * 16, is_valid, new_ctaid);
            mbarrier_arrive_cluster(cta0_clc_empty_addr + clc_stage * 8);
            clc_stage = (clc_stage + 1) % NUM_CLC_STAGES;
            if (clc_stage == 0) clc_full_phase ^= 1;

            if (!is_valid) break;
        }
    }
    // ── scheduler warp ────────────────────────────────────────────────────────
    else if (warp_id == scheduler_warp && elect_sync())
    {
        if (cta_rank == 0)
        {
            #pragma unroll
            for (int s = 0; s < NUM_CLC_STAGES; ++s)
            {
                mbarrier_arrive_expect_cluster(clc_full_mbar_addr + s * 8, 16);
                int remote_full = map_smem_addr_to_cta_rank(clc_full_mbar_addr + s * 8, 1);
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
            clc_stage = (clc_stage + 1) % NUM_CLC_STAGES;
            if (clc_stage == 0) clc_full_phase ^= 1;

            if (!is_valid) break;

            if (cta_rank == 0)
            {
                mbarrier_wait(clc_empty_mbar_addr + issue_stage * 8, clc_empty_phase);
                if (issue_stage == NUM_CLC_STAGES - 1) clc_empty_phase ^= 1;

                mbarrier_arrive_expect_cluster(clc_full_mbar_addr + issue_stage * 8, 16);
                int remote_full = map_smem_addr_to_cta_rank(clc_full_mbar_addr + issue_stage * 8, 1);
                mbarrier_arrive_expect_cluster(remote_full, 16);
                clc_try_cancel(clc_response_addr + issue_stage * 16,
                               clc_full_mbar_addr + issue_stage * 8);
            }
        }
    }
    // ── epilogue warps ────────────────────────────────────────────────────────
    else if (warp_id < NUM_EPI_WARPS)
    {
        constexpr int EPI_BAR     = 7;
        constexpr int EPI_THREADS = NUM_EPI_WARPS * WARP_SIZE;
        constexpr int num_chunks  = BN / STORE_N;
        constexpr int LOADS_PER   = STORE_N / 8;

        int block_col = mn_tile % grid_n;
        int block_row = (mn_tile / grid_n) * CTA_GROUP_SIZE + (int)cta_rank;
        int cur_batch = batch;

        int wave_iter = 0;
        int clc_stage = 0, clc_full_phase = 0;

        while (true)
        {
            const int wave_stage = wave_iter % NUM_EPI_STAGES;
            const int wave_phase = (wave_iter / NUM_EPI_STAGES) & 1;

            mbarrier_wait(tmem_full_addr + wave_stage * 8, wave_phase);
            tcgen05_fence_after();

            // TMEM row: CTA0 → rows [0..BM-1], CTA1 → rows [BM..2*BM-1]
            const int tmem_row = (tmem_addr + wave_stage * BN)
                               + ((int(cta_rank) * BM + warp_id * WARP_SIZE) << 16);
            const int row = warp_id * WARP_SIZE + lane_id;
            int store_stage = 0;

            for (int chunk = 0; chunk < num_chunks; ++chunk)
            {
                if (warp_id == 0) tma_store_wait<NUM_STORE_STGS - 1>();

                float tmp[LOADS_PER][8];
                #pragma unroll
                for (int n = 0; n < LOADS_PER; ++n)
                    tcgen05_ld(tmp[n], tmem_row + chunk * STORE_N + n * 8);
                tcgen05_wait_ld();

                if (chunk == num_chunks - 1)
                {
                    tcgen05_fence_before();
                    // Clear bit 24 to address CTA0's tmem_empty regardless of cta_rank
                    const int tmem_empty_cta0 = (tmem_empty_addr + wave_stage * 8) & 0xFEFFFFFF;
                    if (elect_sync())
                        mbarrier_arrive_cluster(tmem_empty_cta0);
                }

                barrier_sync(EPI_BAR, EPI_THREADS);

                #pragma unroll
                for (int n = 0; n < LOADS_PER; ++n)
                {
                    nv_bfloat162 packed_bf[4];
                    #pragma unroll
                    for (int i = 0; i < 4; ++i)
                    {
                        float v0 = tmp[n][i * 2]     * alpha;
                        float v1 = tmp[n][i * 2 + 1] * alpha;
                        packed_bf[i] = __float22bfloat162_rn({v0, v1});
                    }

                    if (beta != 0.0f)
                    {
                        const int c_col = block_col * BN + chunk * STORE_N + n * 8;
                        const int c_row = cur_batch * M + block_row * BM + row;
                        const __nv_bfloat16* cp = C_in + (long long)c_row * N + c_col;
                        #pragma unroll
                        for (int i = 0; i < 4; ++i)
                        {
                            __nv_bfloat162 old = *reinterpret_cast<const __nv_bfloat162*>(cp + i * 2);
                            packed_bf[i] = __hadd2(packed_bf[i],
                                           __hmul2(old, __float22bfloat162_rn({beta, beta})));
                        }
                    }

                    const int swizzled_n = n ^ (row & 7);
                    __nv_bfloat16* wp = store_base
                                      + store_stage * BM * STORE_N
                                      + row * STORE_N + swizzled_n * 8;
                    *reinterpret_cast<int4*>(wp) = *reinterpret_cast<int4*>(packed_bf);
                }

                __syncwarp();
                tma_store_fence();
                barrier_sync(EPI_BAR, EPI_THREADS);

                if (warp_id == 0 && elect_sync())
                {
                    const int src   = store_smem + store_stage * store_buf_sz;
                    const int c_col = block_col * BN + chunk * STORE_N;
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
            clc_stage = (clc_stage + 1) % NUM_CLC_STAGES;
            if (clc_stage == 0) clc_full_phase ^= 1;

            if (!is_valid) break;

            const int new_tile_idx = (int)(new_ctaid / CTA_GROUP_SIZE);
            cur_batch  = new_tile_idx / total_mn_tiles;
            const int new_mn = new_tile_idx % total_mn_tiles;
            block_col = new_mn % grid_n;
            block_row = (new_mn / grid_n) * CTA_GROUP_SIZE + (int)cta_rank;
        }

        if (warp_id == 0) tma_store_wait<0>();
        barrier_sync(EPI_BAR, EPI_THREADS);
    }

    cluster_sync();
    if (warp_id == 0)
        dealloc_tmem<CTA_GROUP_SIZE>(tmem_addr, BN * NUM_EPI_STAGES);
}

// ─── launcher ─────────────────────────────────────────────────────────────────

template<int BM, int BN, int BK>
static void launch_bgemm_nn_cluster(
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
                "[BluBridge SM103 NN Cluster] WARNING: device '%s' SM %d.%d, requires SM 10.3+\n",
                prop.name, prop.major, prop.minor);
    }
    if (!s_arch_ok) return;

    // Compute smem budget
    constexpr int a_tile    = BM * BK * (int)sizeof(__nv_bfloat16);
    constexpr int b_tile    = (BN / CTA_GROUP_SIZE) * BK * (int)sizeof(__nv_bfloat16);
    constexpr int tile_size = a_tile + b_tile;
    constexpr int store_tot = NUM_STORE_STGS * BM * STORE_N * (int)sizeof(__nv_bfloat16);
    constexpr int smem_budget = 227 * 1024;
    constexpr int QUEUE_SIZE  = (smem_budget - store_tot) / tile_size;
    static_assert(QUEUE_SIZE >= 2, "QUEUE_SIZE too small — reduce BM/BN/BK");
    constexpr int smem_size = QUEUE_SIZE * tile_size + store_tot;

    // Build TMA maps
    CUtensorMap A_tmap, B_tmap, C_tmap;
    init_4d_tma_A(&A_tmap, A, M, K, BM, BK,
                  (uint64_t)batchCount, (uint64_t)M * K,
                  CU_TENSOR_MAP_SWIZZLE_128B);
    init_4d_tma_B_NN(&B_tmap, B, K, N, BK, BN,
                     (uint64_t)batchCount, (uint64_t)K * N,
                     CU_TENSOR_MAP_SWIZZLE_128B);
    init_2d_tma_C(&C_tmap, C,
                  (uint64_t)batchCount * M, (uint64_t)N,
                  (uint32_t)BM, (uint32_t)STORE_N,
                  CU_TENSOR_MAP_SWIZZLE_128B);

    // Tile grid dimensions (no GPU-side tile_map needed — computed on-device)
    const int grid_m         = M / (CTA_GROUP_SIZE * BM); // pairs of BM-row blocks
    const int grid_n         = N / BN;
    const int total_mn_tiles = grid_m * grid_n;

    auto kernel = bgemm_sm103_bf16_nn_cluster_kernel<BM, BN, BK, QUEUE_SIZE>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    const int total_tiles = total_mn_tiles * batchCount;
    dim3 grid(total_tiles * CTA_GROUP_SIZE);
    // Block size scales with BM: (BM/32 epilogue + 3 utility) warps
    const int block_size = (BM / WARP_SIZE + NUM_PROD_WARPS + NUM_CONS_WARPS + NUM_SCHED_WARPS) * WARP_SIZE;

    kernel<<<grid, block_size, smem_size, stream>>>(
        A, C, C,
        M, N, K, total_mn_tiles,
        alpha, beta,
        A_tmap, B_tmap, C_tmap,
        grid_n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "[BluBridge SM103 NN Cluster] Launch error: %s\n",
                cudaGetErrorString(err));
}

// ─── multi-config dispatcher (ThunderKittens-style) ──────────────────────────
// Selects the best (BM, BN) tile based on problem shape, mirroring how
// ThunderKittens picks Nb=64 for small N and Nb=256 for large N:
//   BM=64  when M < 2×128 (cluster covers 2×64=128 M-rows — handles M=128)
//   BN=256 when N is wide (≥256, divisible by 256 — wider tile, better AI)
//   BM=128,BN=128 otherwise (safe default)

static void dispatch_bgemm_nn_cluster(
    const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
    int M, int N, int K, int batchCount, float alpha, float beta, cudaStream_t stream)
{
    // Only the BM=128, BN=128 tile has been validated. The dispatcher in
    // Bgemm_sm103_dispatcher.cu already guards M%256==0 before reaching here.
    launch_bgemm_nn_cluster<128, 128, 64>(A, B, C, M, N, K, batchCount, alpha, beta, stream);
}

// ─── public C API ─────────────────────────────────────────────────────────────

extern "C" void mycublasBgemmSM103_bf16_nn_cluster_128x128x64(
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
    dispatch_bgemm_nn_cluster(A, B, C, M, N, K, batchCount, alpha, beta, stream);
}

extern "C" void mycublasBgemmSM103_bf16_nn_cluster_256x256x64(
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
    // BM=256 needs M divisible by 2×256=512; fall back to the smarter dispatcher otherwise
    if (M % (CTA_GROUP_SIZE * 256) == 0)
        launch_bgemm_nn_cluster<256, 256, 64>(A, B, C, M, N, K, batchCount, alpha, beta, stream);
    else
        dispatch_bgemm_nn_cluster(A, B, C, M, N, K, batchCount, alpha, beta, stream);
}
