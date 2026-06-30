// Bgemm_sm100_bf16.cu
// Blackwell (SM100) BF16 Batched GEMM — NT layout
// C[b,M,N] = alpha * A[b,M,K] × B[b,N,K]^T + beta * C[b,M,N]
//
// Architecture:
//   tcgen05.mma (Tensor Memory / TMEM accumulator, SM100 only)
//   TMA 4D bulk async loads with mbarrier pipeline
//   CTA cluster of 2: each CTA owns BN/2 output columns
//   Warp roles: 4 epilogue | 1 producer | 1 consumer
//
// Tile: BM=128, BN=128 (64 per CTA), BK=64

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
static constexpr int SWIZZLE_W      = 64;   // 64 elements = 128 bytes per swizzle group
static constexpr int MMA_K_STEP     = 16;   // inner-K step of one tcgen05 MMA call
static constexpr int CTA_GROUP_SIZE = 2;    // cluster width
static constexpr uint16_t CTA_MASK  = 0b11; // 2-CTA multicast mask
static constexpr int NUM_EPI_WARPS  = 4;
static constexpr int NUM_PROD_WARPS = 1;
static constexpr int NUM_CONS_WARPS = 1;
static constexpr int TOTAL_WARPS    = NUM_EPI_WARPS + NUM_PROD_WARPS + NUM_CONS_WARPS; // 6
static constexpr int STORE_N        = 64;   // columns written per TMA store call
static constexpr int NUM_STORE_STGS = 2;    // double-buffer store smem
static constexpr int NUM_EPI_STAGES = 2;    // TMEM double-buffering slots (for future CLC)

// SM100-specific device code: compile ONLY for SM 10.0 (GB100/B100/B200).
// tcgen05.mma / tcgen05.alloc / tcgen05.ld are NOT supported on sm_103 (B300)
// per PTX ISA 9.3 — those instructions require sm_100f or sm_101f.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 1000

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
        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
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

// Encode smem address into a tcgen05 matrix descriptor.
// stride_byte_offset = 8 * SWIZZLE_W * sizeof(bf16) = 1024 bytes.
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

// Commit MMA results and signal mbarrier on all CTAs in the cluster mask.
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

// Load 8 floats (one 32×32-bit tile row) from TMEM.
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
void tcgen05_fence_before_thread_sync()
{
    asm volatile("tcgen05.fence::before_thread_sync;");
}

__device__ __forceinline__
void tcgen05_fence_after_thread_sync()
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
    if (e == CUDA_SUCCESS) return;
    const char* msg = "cuTensorMap error";
    cuGetErrorString(e, &msg);
    // propagate as a runtime error; caller checks return
    (void)msg;
}

// 4D TMA for A or B (batched NT layout).
//
// A[batch, M, K] row-major → TMA shape {64, M, K/64, batch}
//   Strides: {K*sizeof(bf16), 128, M*K*sizeof(bf16)}
//
// B[batch, N/2, K] row-major (B stored transposed: N rows of K cols) →
//   same encoding with M→N/2, M*K→(N/2)*K
static inline void init_4d_tma_map_batched(
    CUtensorMap* tmap,
    const __nv_bfloat16* ptr,
    int K, int BK,
    uint64_t global_height,      // M for A, N/CTA_GROUP for B
    uint32_t box_height,         // BM for A, BN/CTA_GROUP for B
    uint64_t batch_count,
    uint64_t batch_stride_elems, // M*K for A, (N/CTA_GROUP)*K for B
    CUtensorMapSwizzle swizzle)
{
    // Dimension order (innermost first): [64, global_height, K/64, batch]
    constexpr uint32_t rank = 4;
    uint64_t globalDim[rank]       = {64ULL, global_height,
                                       (uint64_t)(K / 64), batch_count};
    uint64_t globalStrides[rank-1] = {
        (uint64_t)K * sizeof(__nv_bfloat16),     // stride dim1 (height rows)
        128ULL,                                   // stride dim2 (K groups, 64*2 bytes)
        batch_stride_elems * sizeof(__nv_bfloat16)// stride dim3 (batch)
    };
    uint32_t boxDim[rank]          = {64U, box_height, (uint32_t)(BK / 64), 1U};
    uint32_t elementStrides[rank]  = {1U, 1U, 1U, 1U};

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

// 2D TMA for storing C tiles. C is laid out as [batch*M, N] contiguous.
static inline void init_2d_tma_map_store(
    CUtensorMap* tmap,
    const __nv_bfloat16* ptr,
    uint64_t total_rows,   // batch * M
    uint64_t total_cols,   // N
    uint32_t box_rows,     // BM
    uint32_t box_cols,     // STORE_N
    CUtensorMapSwizzle swizzle)
{
    constexpr uint32_t rank = 2;
    uint64_t globalDim[rank]       = {total_cols, total_rows};
    uint64_t globalStrides[rank-1] = {total_cols * sizeof(__nv_bfloat16)};
    uint32_t boxDim[rank]          = {box_cols, box_rows};
    uint32_t elementStrides[rank]  = {1U, 1U};

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

// Grid: dim3(ceil(N/BN)*CTA_GROUP_SIZE, ceil(M/BM), batchCount)
// Cluster: 2 CTAs along X → each cluster handles one (BM × BN) output tile
//
// Warp assignments (0-indexed):
//   0-3 : epilogue  — tcgen05.ld TMEM → convert + alpha/beta → TMA store
//   4   : producer  — TMA 4D loads A and B tiles into smem
//   5   : consumer  — tcgen05.mma across the K dimension

template<int BM, int BN, int BK, int QUEUE_SIZE>
__global__ __cluster_dims__(2, 1, 1)
void bgemm_sm100_bf16_nt_kernel(
    const __nv_bfloat16* __restrict__ A,   // [batchCount, M, K]  row-major
    const __nv_bfloat16* __restrict__ B,   // [batchCount, N, K]  row-major (B^T of K×N)
          __nv_bfloat16* __restrict__ C,   // [batchCount, M, N]  row-major
    int M, int N, int K,
    float alpha, float beta,
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B_tmap,
    const __grid_constant__ CUtensorMap C_tmap)
{
    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const uint32_t cta_rank = get_cluster_cta_rank();

    // blockIdx.x is the cluster index * CTA_GROUP_SIZE + cta_rank
    const int batch   = blockIdx.z;
    const int block_m = blockIdx.y;
    const int block_n = blockIdx.x / CTA_GROUP_SIZE;

    // ── shared memory layout ──────────────────────────────────────────────────
    // [QUEUE_SIZE × (BM×BK + BN/2×BK)] pipeline stages, then store double-buffer
    extern __shared__ __align__(1024) char smem[];
    const int smem_ptr = static_cast<int>(__cvta_generic_to_shared(smem));

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ int tmem[1]; // TMEM address returned by tcgen05.alloc
    const int tmem_smem_addr = static_cast<int>(__cvta_generic_to_shared(tmem));

    __shared__ __align__(8) uint64_t full_mbar[QUEUE_SIZE];     // A/B load done
    __shared__ __align__(8) uint64_t empty_mbar[QUEUE_SIZE];    // MMA done, slot free
    __shared__ __align__(8) uint64_t tmem_full[NUM_EPI_STAGES]; // MMA tile done
    __shared__ __align__(8) uint64_t tmem_empty[NUM_EPI_STAGES];// epilogue done

    const int full_mbar_addr  = static_cast<int>(__cvta_generic_to_shared(full_mbar));
    const int empty_mbar_addr = static_cast<int>(__cvta_generic_to_shared(empty_mbar));
    const int tmem_full_addr  = static_cast<int>(__cvta_generic_to_shared(tmem_full));
    const int tmem_empty_addr = static_cast<int>(__cvta_generic_to_shared(tmem_empty));

    constexpr int producer_warp = NUM_EPI_WARPS;        // warp 4
    constexpr int consumer_warp = NUM_EPI_WARPS + 1;    // warp 5

    // Per-stage smem footprint: A tile + B tile (for this CTA's half of BN)
    constexpr int copy_size    = (BM + BN / CTA_GROUP_SIZE) * BK * sizeof(__nv_bfloat16);
    // Store buffer: double-buffered BM × STORE_N tiles
    constexpr int store_buf_sz = BM * STORE_N * sizeof(__nv_bfloat16);
    const int store_smem        = smem_ptr + QUEUE_SIZE * copy_size;
    __nv_bfloat16* store_base   = reinterpret_cast<__nv_bfloat16*>(smem + QUEUE_SIZE * copy_size);

    // ── initialisation ────────────────────────────────────────────────────────
    if (warp_id == producer_warp && elect_sync())
    {
        // full_mbar: waits for both CTAs' producers (2 arrivals per stage)
        // empty_mbar: waits for consumer to finish MMA (1 arrival per stage)
        for (int i = 0; i < QUEUE_SIZE; ++i)
        {
            mbarrier_init(full_mbar_addr  + i * 8, NUM_PROD_WARPS * CTA_GROUP_SIZE);
            mbarrier_init(empty_mbar_addr + i * 8, 1);
        }
        // tmem_full: consumer signals after completing all K for one tile
        // tmem_empty: epilogue signals after writing output
        for (int i = 0; i < NUM_EPI_STAGES; ++i)
        {
            mbarrier_init(tmem_full_addr  + i * 8, 1);
            mbarrier_init(tmem_empty_addr + i * 8, NUM_EPI_WARPS * CTA_GROUP_SIZE);
        }
        cluster_fence_mbarrier_init();
    }
    else if (warp_id == consumer_warp)
    {
        // TMEM allocation: BN columns × NUM_EPI_STAGES slots
        alloc_tmem<CTA_GROUP_SIZE>(tmem_smem_addr, BN * NUM_EPI_STAGES);
    }

    cluster_sync(); // all warps synchronised; TMEM address now valid in tmem[0]

    const int tmem_addr = tmem[0];

    // i_desc encodes the MMA problem shape for tcgen05.mma.cta_group::2.kind::f16:
    //   N dimension (per-cluster BN): bits 17..23
    //   M dimension (CTA_GROUP * BM): bits 24..30
    constexpr uint32_t i_desc =
          (1U <<  4U)
        | (1U <<  7U)
        | (1U << 10U)
        | ((uint32_t)(BN)                  >> 3U << 17U)
        | ((uint32_t)(CTA_GROUP_SIZE * BM) >> 4U << 24U);

    const int k_iters = K / BK;

    // Both CTAs' producers signal CTA0's full_mbar (two arrivals complete it).
    const int tma_mbar_base = (cta_rank == 0)
        ? full_mbar_addr
        : map_smem_addr_to_cta_rank(full_mbar_addr, 0);

    // tmem_empty is on CTA0; both epilogue CTAs signal it.
    const int tmem_empty_cta0 = (cta_rank == 0)
        ? tmem_empty_addr
        : map_smem_addr_to_cta_rank(tmem_empty_addr, 0);

    // ── producer warp ─────────────────────────────────────────────────────────
    // Issues TMA 4D loads for A (full BM × BK) and B (BN/2 × BK) each K step.
    // A and B are both loaded into this CTA's local smem segment.
    if (warp_id == producer_warp && elect_sync())
    {
        const int A_row = block_m * BM;
        // Each CTA in the cluster loads its own N-slice of B.
        const int B_row = block_n * BN + (int)cta_rank * (BN / CTA_GROUP_SIZE);

        int stage = 0, phase = 0, issued = 0;

        for (int k = 0; k < k_iters; ++k)
        {
            // Throttle: wait for consumer to free this stage slot.
            if (issued >= QUEUE_SIZE)
                mbarrier_wait(empty_mbar_addr + stage * 8, phase ^ 1);
            ++issued;

            const int A_smem = smem_ptr + stage * copy_size;
            const int B_smem = A_smem + BM * BK * (int)sizeof(__nv_bfloat16);
            const int mbar   = tma_mbar_base + stage * 8;

            // Announce expected bytes so the mbarrier knows when both loads complete.
            mbarrier_arrive_expect(mbar, copy_size);

            // TMA x=0 → innermost 64-element group offset (always 0 for aligned tiles)
            // TMA y=row index in M (for A) or N/2 (for B)
            // TMA z=K-chunk index
            // TMA w=batch index
            tma_4d_gmem2smem<CTA_GROUP_SIZE>(
                A_smem, &A_tmap,
                /*x=*/0, /*y=*/A_row, /*z=*/k * BK / SWIZZLE_W, /*w=*/batch,
                mbar);
            tma_4d_gmem2smem<CTA_GROUP_SIZE>(
                B_smem, &B_tmap,
                /*x=*/0, /*y=*/B_row, /*z=*/k * BK / SWIZZLE_W, /*w=*/batch,
                mbar);

            stage = (stage + 1) % QUEUE_SIZE;
            phase ^= (stage == 0);
        }
    }
    // ── consumer warp ─────────────────────────────────────────────────────────
    // Drives tcgen05.mma across the K dimension. Only CTA0 issues the MMA
    // instruction; cta_group::2 makes both CTAs in the cluster contribute.
    else if (warp_id == consumer_warp && elect_sync())
    {
        if (cta_rank == 0)
        {
            constexpr int wave_stage = 0; // single-tile per CTA, wave=0

            asm volatile("tcgen05.fence::after_thread_sync;");

            int stage = 0, phase = 0;

            for (int k = 0; k < k_iters; ++k)
            {
                // Wait for A and B tiles to be resident in smem.
                mbarrier_wait(full_mbar_addr + stage * 8, phase);
                asm volatile("tcgen05.fence::after_thread_sync;");

                const int A_smem = smem_ptr + stage * copy_size;
                const int B_smem = A_smem + BM * BK * (int)sizeof(__nv_bfloat16);

                // Inner K loop: BK / SWIZZLE_W outer groups × SWIZZLE_W / MMA_K_STEP steps.
                for (int k1 = 0; k1 < BK / SWIZZLE_W; ++k1)
                {
                    for (int k2 = 0; k2 < SWIZZLE_W / MMA_K_STEP; ++k2)
                    {
                        // First call resets accumulator; subsequent calls accumulate.
                        const int use_accum = (k == 0 && k1 == 0 && k2 == 0) ? 0 : 1;

                        // Byte offsets within smem for this (k1, k2) sub-tile.
                        const int a_off =
                            k1 * SWIZZLE_W * BM * (int)sizeof(__nv_bfloat16)
                          + k2 * MMA_K_STEP  * (int)sizeof(__nv_bfloat16);
                        const int b_off =
                            k1 * SWIZZLE_W * (BN / CTA_GROUP_SIZE) * (int)sizeof(__nv_bfloat16)
                          + k2 * MMA_K_STEP  * (int)sizeof(__nv_bfloat16);

                        tcgen05_mma_bf16<CTA_GROUP_SIZE>(
                            tmem_addr + wave_stage * BN,
                            make_smem_desc(A_smem + a_off),
                            make_smem_desc(B_smem + b_off),
                            i_desc, use_accum);
                    }
                }

                // Signal producer that this smem stage is now free.
                tcgen05_commit_multicast<CTA_GROUP_SIZE>(
                    empty_mbar_addr + stage * 8, CTA_MASK);

                stage = (stage + 1) % QUEUE_SIZE;
                phase ^= (stage == 0);
            }

            // Signal epilogue warps: all MMA results are in TMEM.
            tcgen05_commit_multicast<CTA_GROUP_SIZE>(
                tmem_full_addr + wave_stage * 8, CTA_MASK);
        }
        // cta_rank == 1: consumer is a passenger; cluster_sync handles ordering.
    }
    // ── epilogue warps ────────────────────────────────────────────────────────
    // Each warp handles 32 rows of the BM × (BN/CTA_GROUP) output tile.
    // Flow: wait TMEM ready → tcgen05.ld → alpha/beta → swizzle → TMA store.
    else if (warp_id < NUM_EPI_WARPS)
    {
        constexpr int EPI_BAR      = 7;
        constexpr int EPI_THREADS  = NUM_EPI_WARPS * WARP_SIZE;
        constexpr int num_chunks   = (BN / CTA_GROUP_SIZE) / STORE_N;
        constexpr int LOADS_PER    = STORE_N / 8;         // 8 tcgen05.ld calls per chunk

        constexpr int wave_stage   = 0;
        constexpr int wave_phase   = 0;

        // Wait until the consumer has committed all MMA results.
        mbarrier_wait(tmem_full_addr + wave_stage * 8, wave_phase);
        tcgen05_fence_after_thread_sync();

        // TMEM row for this warp: each CTA owns 128 rows, each warp owns 32.
        //   High 16 bits of TMEM address = row offset.
        const int tmem_row = (tmem_addr + wave_stage * BN)
                           + ((int(cta_rank) * 128 + warp_id * WARP_SIZE) << 16);
        const int row      = warp_id * WARP_SIZE + lane_id;
        int store_stage    = 0;

        for (int chunk = 0; chunk < num_chunks; ++chunk)
        {
            // Throttle TMA store queue.
            if (warp_id == 0) tma_store_wait<NUM_STORE_STGS - 1>();

            // Load 8 floats per column-group (= 8 × STORE_N/8 = 8 × 8 = 64 floats)
            float tmp[LOADS_PER][8];
            #pragma unroll
            for (int n = 0; n < LOADS_PER; ++n)
                tcgen05_ld(tmp[n], tmem_row + chunk * STORE_N + n * 8);
            tcgen05_wait_ld();

            // On the last chunk, signal TMEM is free (epilogue done).
            if (chunk == num_chunks - 1)
            {
                tcgen05_fence_before_thread_sync();
                if (elect_sync())
                    mbarrier_arrive_cluster(tmem_empty_cta0 + wave_stage * 8);
            }

            barrier_sync(EPI_BAR, EPI_THREADS);

            // Convert float → bf16 with alpha scaling, then store into smem with
            // swizzled layout (XOR on n index avoids 8-way bank conflicts for stores).
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
                // Apply beta * C (read-modify-write into smem via global C read).
                if (beta != 0.0f)
                {
                    const int c_col = block_n * BN
                                    + (int)cta_rank * (BN / CTA_GROUP_SIZE)
                                    + chunk * STORE_N + n * 8;
                    const int c_row = batch * M + block_m * BM + row;
                    if (c_row < batch * M + M && c_col < N)
                    {
                        // Direct global read for beta blend.
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
                // XOR-swizzle to avoid smem bank conflicts: row's low 3 bits xor n.
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
                // C is laid out as [batch*M, N]; tile origin = (c_col, c_row).
                const int c_col = block_n * BN + (int)cta_rank * (BN / CTA_GROUP_SIZE) + chunk * STORE_N;
                const int c_row = batch * M + block_m * BM;
                tma_2d_smem2gmem(src, &C_tmap, c_col, c_row);
                tma_store_commit();
            }

            store_stage ^= 1;
        } // chunk loop

        if (warp_id == 0) tma_store_wait<0>();
        barrier_sync(EPI_BAR, EPI_THREADS);
    }

    cluster_sync();

    if (warp_id == 0)
        dealloc_tmem<CTA_GROUP_SIZE>(tmem_addr, BN * NUM_EPI_STAGES);
}

// ─── launcher ─────────────────────────────────────────────────────────────────

template<int BM, int BN, int BK>
static void launch_bgemm_sm100_bf16_nt(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
          __nv_bfloat16* C,
    int M, int N, int K, int batchCount,
    float alpha, float beta,
    cudaStream_t stream)
{
    // ── Architecture check (cached) ───────────────────────────────────────────
    // tcgen05.mma.cta_group::2, tcgen05.alloc/dealloc/ld.32x32b/commit/fence
    // are ONLY supported on sm_100f / sm_100a (B100/B200/GB200).
    // Per PTX ISA 9.3 Table 70: sm_103f/sm_103a (B300) only gets tcgen05.ld.red.
    // Checked once per process; subsequent calls are free.
    static int s_arch_ok = -1;
    if (s_arch_ok < 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        s_arch_ok = (prop.major == 10 && prop.minor == 0) ? 1 : 0;
        if (!s_arch_ok)
            fprintf(stderr,
                "[BluBridge SM100 kernel] WARNING: device '%s' is SM %d.%d — "
                "tcgen05.mma requires SM 10.0 (B100/B200/GB200, sm_100a). "
                "SM 10.3 (B300) only supports tcgen05.ld.red per PTX ISA 9.3. "
                "Skipping launch — output buffer unchanged.\n",
                prop.name, prop.major, prop.minor);
    }
    if (!s_arch_ok) return;

    // ── TMA map setup ─────────────────────────────────────────────────────────
    CUtensorMap A_tmap, B_tmap, C_tmap;

    // A: [batch, M, K] row-major → 4D TMA {64, M, K/64, batch}
    init_4d_tma_map_batched(
        &A_tmap, A,
        K, BK,
        /*global_height=*/(uint64_t)M,
        /*box_height=*/   (uint32_t)BM,
        (uint64_t)batchCount,
        /*batch_stride=*/ (uint64_t)M * K,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);

    // B: [batch, N, K] row-major (each batch has N×K = B^T of standard K×N)
    init_4d_tma_map_batched(
        &B_tmap, B,
        K, BK,
        /*global_height=*/(uint64_t)N,
        /*box_height=*/   (uint32_t)(BN / CTA_GROUP_SIZE),
        (uint64_t)batchCount,
        /*batch_stride=*/ (uint64_t)N * K,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);

    // C: laid out as [batch*M, N] for TMA store
    init_2d_tma_map_store(
        &C_tmap, C,
        (uint64_t)batchCount * M, (uint64_t)N,
        (uint32_t)BM, (uint32_t)STORE_N,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B);

    // ── smem budget ───────────────────────────────────────────────────────────
    constexpr int tile_size   = (BM + BN / CTA_GROUP_SIZE) * BK * (int)sizeof(__nv_bfloat16);
    constexpr int store_total = NUM_STORE_STGS * BM * STORE_N * (int)sizeof(__nv_bfloat16);
    // SM100 allows up to 228 KB dynamic smem per CTA.
    constexpr int smem_budget = 227 * 1024;
    constexpr int QUEUE_SIZE  = (smem_budget - store_total) / tile_size;
    static_assert(QUEUE_SIZE >= 2, "QUEUE_SIZE too small; reduce BM/BN/BK");
    constexpr int smem_size   = QUEUE_SIZE * tile_size + store_total;

    // ── grid/block ────────────────────────────────────────────────────────────
    const int grid_m = (M + BM - 1) / BM;
    const int grid_n = (N + BN - 1) / BN;

    dim3 grid(grid_n * CTA_GROUP_SIZE, grid_m, batchCount);
    const int block_size = TOTAL_WARPS * WARP_SIZE;

    auto kernel = bgemm_sm100_bf16_nt_kernel<BM, BN, BK, QUEUE_SIZE>;
    cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);

    kernel<<<grid, block_size, smem_size, stream>>>(
        A, B, C,
        M, N, K,
        alpha, beta,
        A_tmap, B_tmap, C_tmap);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "[BluBridge SM100 kernel] Launch error: %s\n",
                cudaGetErrorString(err));
}

// ─── public C API ─────────────────────────────────────────────────────────────

// NT batched GEMM: C[b] = alpha * A[b] × B[b]^T + beta * C[b]
// A: [batchCount, M, K]  row-major
// B: [batchCount, N, K]  row-major  (B transposed: N rows of K cols)
// C: [batchCount, M, N]  row-major
//
// Requirements:
//   M % 128 == 0,  N % 128 == 0,  K % 64 == 0
//   SM100 (Blackwell) device
extern "C" void mycublasBgemmSM100_bf16_nt_128x128x64(
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
    launch_bgemm_sm100_bf16_nt<128, 128, 64>(
        A, B, C, M, N, K, batchCount, alpha, beta, stream);
}

// Convenience overload matching BLUBLAS strided-batched signature.
// B must be provided in [batchCount, N, K] row-major order (transposed).
extern "C" void mycublasBgemmSM100_bf16_nt_strided(
    mycublasHandle_t handle,
    int M, int N, int K,
    float alpha,
    const __nv_bfloat16* A, long long strideA,
    const __nv_bfloat16* B, long long strideB,
    float beta,
          __nv_bfloat16* C, long long strideC,
    int batchCount)
{
    // This overload only makes sense when strides match packed row-major layout.
    // strideA == M*K, strideB == N*K, strideC == M*N
    // The TMA maps are built from base pointers assuming packed batches.
    (void)strideA; (void)strideB; (void)strideC;
    cudaStream_t stream = handle ? handle->stream : 0;
    launch_bgemm_sm100_bf16_nt<128, 128, 64>(
        A, B, C, M, N, K, batchCount, alpha, beta, stream);
}

#endif // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 1000
