// #include "mycublas.h"
// #include "Sgemm_core_template_sm89.cuh"

// extern "C" void get_gpu_info(int *sm_ver, int *sm_count);

// // SMEM Scale/Prep Kernel for SplitK
// extern "C" void launch_sgemm_sm89_scale(float* C, float beta, int M, int N, int ldc, long long strideC, int batchCount, cudaStream_t stream) {
//     if (batchCount == 0) return;
//     dim3 block(32, 32);
//     dim3 grid((N + 31) / 32, (M + 31) / 32, batchCount);
//     sgemm_sm89_scale_kernel<void><<<grid, block, 0, stream>>>(C, beta, M, N, ldc, strideC, batchCount);
// }

// // NN Variants
// extern "C" void launch_sgemm_nn_256x128_bypass_sm89(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount, cudaStream_t stream);
// extern "C" void launch_sgemm_nn_256x128_sm89_template(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount, int splitK, cudaStream_t stream);
// extern "C" void launch_sgemm_nn_128x128_sm89_template(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount, int splitK, cudaStream_t stream);
// extern "C" void launch_sgemm_nn_64x64_sm89_template(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount, int splitK, cudaStream_t stream);
// extern "C" void launch_sgemm_nn_32x32_sm89_template(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount, int splitK, cudaStream_t stream);

// // NT Variants (Template-based)
// extern "C" void launch_sgemm_nt_256x128_sm89_template(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount, int splitK, cudaStream_t stream);
// extern "C" void launch_sgemm_nt_256x128_sm89_2(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount, int splitK, cudaStream_t stream);
// extern "C" void launch_sgemm_nt_128x128_sm89_template(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount, int splitK, cudaStream_t stream);
// extern "C" void launch_sgemm_nt_64x64_sm89_template(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount, int splitK, cudaStream_t stream);
// extern "C" void launch_sgemm_nt_32x32_sm89_template(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount, int splitK, cudaStream_t stream);

// // TN Variants (Template-based)
// extern "C" void launch_sgemm_tn_256x128_sm89_template(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount, int splitK, cudaStream_t stream);
// extern "C" void launch_sgemm_tn_128x128_sm89_template(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount, int splitK, cudaStream_t stream);
// extern "C" void launch_sgemm_tn_64x64_sm89_template(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount, int splitK, cudaStream_t stream);
// extern "C" void launch_sgemm_tn_32x32_sm89_template(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount, int splitK, cudaStream_t stream);
// extern "C" void launch_sgemm_tn_256x64_sm89_2(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount, int splitK, cudaStream_t stream);

// // Tile Configurations for Fallback Template
// using CfgL = SgemmTileConfigSM89<256, 128, 32, 2, 256>;
// using CfgM = SgemmTileConfigSM89<128, 128, 16, 6, 128>;
// using CfgS = SgemmTileConfigSM89<128,  64, 16, 6, 128>;

// template<SgemmLayout Layout, bool IsSplitK>
// void launch_sgemm_sm89_templated(
//     mycublasHandle_t handle, int M, int N, int K,
//     float alpha, const float* d_A, int lda, long long int strideA,
//     const float* d_B, int ldb, long long int strideB,
//     float beta, float* d_C, int ldc, long long int strideC,
//     const float* bias, long long bias_stride,
//     int batchCount, int splitK)
// {
//     if (N <= 64) {
//         constexpr size_t smem_size = CfgS::SMEM_BYTES;
//         static bool cfg = false; if(!cfg){ cudaFuncSetAttribute(sgemm_sm89_kernel<CfgS, true, IsSplitK, Layout>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size); cfg=true; }
//         sgemm_sm89_kernel<CfgS, true, IsSplitK, Layout><<<dim3((N+63)/64, (M+127)/128, batchCount * splitK), 128, smem_size, handle->stream>>>(
//             M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, beta, d_C, ldc, strideC, bias, bias_stride, batchCount, splitK);
//     } else if (M <= 1024 && N <= 1024) {
//         constexpr size_t smem_size = CfgM::SMEM_BYTES;
//         static bool cfg = false; if(!cfg){ cudaFuncSetAttribute(sgemm_sm89_kernel<CfgM, true, IsSplitK, Layout>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size); cfg=true; }
//         sgemm_sm89_kernel<CfgM, true, IsSplitK, Layout><<<dim3((N+127)/128, (M+127)/128, batchCount * splitK), 128, smem_size, handle->stream>>>(
//             M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, beta, d_C, ldc, strideC, bias, bias_stride, batchCount, splitK);
//     } else {
//         constexpr size_t smem_size = CfgL::SMEM_BYTES;
//         static bool cfg = false; if(!cfg){ cudaFuncSetAttribute(sgemm_sm89_kernel<CfgL, true, IsSplitK, Layout>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size); cfg=true; }
//         sgemm_sm89_kernel<CfgL, true, IsSplitK, Layout><<<dim3((N+127)/128, (M+255)/256, batchCount * splitK), 256, smem_size, handle->stream>>>(
//             M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, beta, d_C, ldc, strideC, bias, bias_stride, batchCount, splitK);
//     }
// }

// template<SgemmLayout Layout>
// void launch_sgemm_sm89(
//     mycublasHandle_t handle, int M, int N, int K,
//     float alpha, const float* d_A, int lda, long long int strideA,
//     const float* d_B, int ldb, long long int strideB,
//     float beta, float* d_C, int ldc, long long int strideC,
//     const float* bias, long long bias_stride,
//     int batchCount)
// {
//     launch_sgemm_sm89_templated<Layout, false>(handle, M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, beta, d_C, ldc, strideC, bias, bias_stride, batchCount, 1);
// }

// extern "C" {

// void mycublasSgemmStridedBatched_nn_SM89(
//     mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* d_A, int lda, long long int strideA,
//     const float* d_B, int ldb, long long int strideB, const float beta, float* d_C, int ldc, long long int strideC, int batchCount)
// {
//     int sm_ver, sm_count;
//     get_gpu_info(&sm_ver, &sm_count);
//     if (sm_count <= 0) sm_count = 124; // Fallback

//     // SplitK policy: routes backward grad-W matmuls (K=B*T large, M*N small →
//     // total_blocks < sm_count) through K-striped CTAs with atomicAdd reduction.
//     // Non-deterministic accumulation order biases weight updates by a few ULP;
//     // over 19073 steps this costs ~5.7e-3 val loss vs cuBLAS deterministic
//     // workspace reduction. Acceptable for throughput benchmarks; disable for
//     // full parity runs by forcing splitK=1 below.
//     int splitK = 1;
//     {
//         int blocks_per_batch = ((M + 255) / 256) * ((N + 127) / 128);
//         int total_blocks = blocks_per_batch * batchCount;
//         if (total_blocks * 2 < sm_count && K >= 512) {
//             splitK = min(16, (sm_count + total_blocks - 1) / total_blocks);
//         }
//     }
//     if (splitK > 1 && beta != 1.0f) { launch_sgemm_sm89_scale(d_C, beta, M, N, ldc, strideC, batchCount, handle->stream); }
//     float final_beta = (splitK > 1) ? 1.0f : beta;

//     if (M >= 256 && N >= 128) {
//         launch_sgemm_nn_256x128_sm89_template(M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, final_beta, d_C, ldc, strideC, batchCount, splitK, handle->stream);
//     } else if (M >= 128 && N >= 128) {
//         launch_sgemm_nn_128x128_sm89_template(M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, final_beta, d_C, ldc, strideC, batchCount, splitK, handle->stream);
//     } else if (M >= 64 && N >= 64) {
//         launch_sgemm_nn_64x64_sm89_template(M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, final_beta, d_C, ldc, strideC, batchCount, splitK, handle->stream);
//     } else if (M >= 32 && N >= 32) {
//         launch_sgemm_nn_32x32_sm89_template(M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, final_beta, d_C, ldc, strideC, batchCount, splitK, handle->stream);
//     } else {
//         launch_sgemm_sm89_templated<SgemmLayout::NN, true>(handle, M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, beta, d_C, ldc, strideC, nullptr, 0, batchCount, splitK);
//     }
// }

// void mycublasSgemmStridedBatched_nt_SM89(
//     mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* d_A, int lda, long long int strideA,
//     const float* d_B, int ldb, long long int strideB, const float beta, float* d_C, int ldc, long long int strideC, int batchCount)
// {
//     int sm_ver, sm_count;
//     get_gpu_info(&sm_ver, &sm_count);
//     if (sm_count <= 0) sm_count = 124;

//     // SplitK policy: same rationale as NN dispatcher above. NT is the dominant
//     // grad-W backward path (dW = X^T @ dY, K=B*T=16384, M=N=768 → 18 CTAs on
//     // 142 SMs without splitK = 12.7% utilization). SplitK=8 → 144 CTAs ≈ full
//     // SM saturation. Non-deterministic over full training; acceptable for
//     // throughput benchmarks.
//     int splitK = 1;
//     {
//         int blocks_per_batch = ((M + 255) / 256) * ((N + 127) / 128);
//         int total_blocks = blocks_per_batch * batchCount;
//         if (total_blocks * 2 < sm_count && K >= 512) {
//             splitK = min(16, (sm_count + total_blocks - 1) / total_blocks);
//         }
//     }
//     if (splitK > 1 && beta != 1.0f) { launch_sgemm_sm89_scale(d_C, beta, M, N, ldc, strideC, batchCount, handle->stream); }
//     float final_beta = (splitK > 1) ? 1.0f : beta;

//     if (M >= 256 && N >= 128) {
//         launch_sgemm_nt_256x128_sm89_2(M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, final_beta, d_C, ldc, strideC, batchCount, splitK, handle->stream);
//     } else if (M >= 128 && N >= 128) {
//         launch_sgemm_nt_128x128_sm89_template(M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, final_beta, d_C, ldc, strideC, batchCount, splitK, handle->stream);
//     } else if (M >= 64 && N >= 64) {
//         launch_sgemm_nt_64x64_sm89_template(M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, final_beta, d_C, ldc, strideC, batchCount, splitK, handle->stream);
//     } else if (M >= 32 && N >= 32) {
//         launch_sgemm_nt_32x32_sm89_template(M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, final_beta, d_C, ldc, strideC, batchCount, splitK, handle->stream);
//     } else {
//         launch_sgemm_sm89_templated<SgemmLayout::NT, true>(handle, M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, beta, d_C, ldc, strideC, nullptr, 0, batchCount, splitK);
//     }
// }

// void mycublasSgemmStridedBatched_tn_SM89(
//     mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* d_A, int lda, long long int strideA,
//     const float* d_B, int ldb, long long int strideB, const float beta, float* d_C, int ldc, long long int strideC, int batchCount)
// {
//     int sm_ver, sm_count;
//     get_gpu_info(&sm_ver, &sm_count);
//     if (sm_count <= 0) sm_count = 124;

//     // SplitK policy: same rationale as NN/NT dispatchers. TN backward dW
//     // (M=N=768, K=16384) produces only 18 CTAs at 256×128 tile on 142 SMs.
//     // SplitK restores full utilization. Non-deterministic over full training.
//     int splitK = 1;
//     {
//         int blocks_per_batch = ((M + 255) / 256) * ((N + 127) / 128);
//         int total_blocks = blocks_per_batch * batchCount;
//         if (total_blocks * 2 < sm_count && K >= 512) {
//             splitK = min(16, (sm_count + total_blocks - 1) / total_blocks);
//         }
//     }
//     if (splitK > 1 && beta != 1.0f) { launch_sgemm_sm89_scale(d_C, beta, M, N, ldc, strideC, batchCount, handle->stream); }
//     float final_beta = (splitK > 1) ? 1.0f : beta;

//     if (M >= 256 && N >= 128) {
//         launch_sgemm_tn_256x128_sm89_template(M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, final_beta, d_C, ldc, strideC, batchCount, splitK, handle->stream);
//     } else if (M >= 128 && N >= 128) {
//         launch_sgemm_tn_128x128_sm89_template(M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, final_beta, d_C, ldc, strideC, batchCount, splitK, handle->stream);
//     } else if (M >= 64 && N >= 64) {
//         launch_sgemm_tn_64x64_sm89_template(M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, final_beta, d_C, ldc, strideC, batchCount, splitK, handle->stream);
//     } else if (M >= 32 && N >= 32) {
//         launch_sgemm_tn_32x32_sm89_template(M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, final_beta, d_C, ldc, strideC, batchCount, splitK, handle->stream);
//     } else {
//         launch_sgemm_sm89_templated<SgemmLayout::TN, true>(handle, M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, beta, d_C, ldc, strideC, nullptr, 0, batchCount, splitK);
//     }
// }

// } // extern "C"

























#include "mycublas.h"
#include "Sgemm_core_template_sm89.cuh"

extern "C" void get_gpu_info(int *sm_ver, int *sm_count);

// SMEM Scale/Prep Kernel for SplitK
extern "C" void launch_sgemm_sm89_scale(float* C, float beta, int M, int N, int ldc, long long strideC, int batchCount, cudaStream_t stream) {
    if (batchCount == 0) return;
    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (M + 31) / 32, batchCount);
    sgemm_sm89_scale_kernel<void><<<grid, block, 0, stream>>>(C, beta, M, N, ldc, strideC, batchCount);
}

// NN Variants
extern "C" void launch_sgemm_nn_256x128_bypass_sm89(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount, cudaStream_t stream);
extern "C" void launch_sgemm_nn_256x128_sm89_template(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount, int splitK, cudaStream_t stream);
extern "C" void launch_sgemm_nn_128x128_sm89_template(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount, int splitK, cudaStream_t stream);
extern "C" void launch_sgemm_nn_64x64_sm89_template(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount, int splitK, cudaStream_t stream);
extern "C" void launch_sgemm_nn_32x32_sm89_template(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount, int splitK, cudaStream_t stream);

// NT Variants (Template-based)
extern "C" void launch_sgemm_nt_256x128_sm89_template(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount, int splitK, cudaStream_t stream);
extern "C" void launch_sgemm_nt_256x128_sm89_2(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount, int splitK, cudaStream_t stream);
extern "C" void launch_sgemm_nt_128x128_sm89_template(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount, int splitK, cudaStream_t stream);
extern "C" void launch_sgemm_nt_64x64_sm89_template(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount, int splitK, cudaStream_t stream);
extern "C" void launch_sgemm_nt_32x32_sm89_template(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount, int splitK, cudaStream_t stream);

// TN Variants (Template-based)
extern "C" void launch_sgemm_tn_256x128_sm89_template(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount, int splitK, cudaStream_t stream);
extern "C" void launch_sgemm_tn_128x128_sm89_template(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount, int splitK, cudaStream_t stream);
extern "C" void launch_sgemm_tn_64x64_sm89_template(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount, int splitK, cudaStream_t stream);
extern "C" void launch_sgemm_tn_32x32_sm89_template(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount, int splitK, cudaStream_t stream);
extern "C" void launch_sgemm_tn_256x64_sm89_2(int M, int N, int K, float alpha, const float* A, int lda, long long strideA, const float* B, int ldb, long long strideB, float beta, float* C, int ldc, long long strideC, int batchCount, int splitK, cudaStream_t stream);

// Tile Configurations for Fallback Template
using CfgL = SgemmTileConfigSM89<256, 128, 32, 2, 256>;
using CfgM = SgemmTileConfigSM89<128, 128, 16, 6, 128>;
using CfgS = SgemmTileConfigSM89<128,  64, 16, 6, 128>;

template<SgemmLayout Layout, bool IsSplitK>
void launch_sgemm_sm89_templated(
    mycublasHandle_t handle, int M, int N, int K,
    float alpha, const float* d_A, int lda, long long int strideA,
    const float* d_B, int ldb, long long int strideB,
    float beta, float* d_C, int ldc, long long int strideC,
    const float* bias, long long bias_stride,
    int batchCount, int splitK)
{
    if (N <= 64) {
        constexpr size_t smem_size = CfgS::SMEM_BYTES;
        static bool cfg = false; if(!cfg){ cudaFuncSetAttribute(sgemm_sm89_kernel<CfgS, true, IsSplitK, Layout>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size); cfg=true; }
        sgemm_sm89_kernel<CfgS, true, IsSplitK, Layout><<<dim3((N+63)/64, (M+127)/128, batchCount * splitK), 128, smem_size, handle->stream>>>(
            M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, beta, d_C, ldc, strideC, bias, bias_stride, batchCount, splitK);
    } else if (M <= 1024 && N <= 1024) {
        constexpr size_t smem_size = CfgM::SMEM_BYTES;
        static bool cfg = false; if(!cfg){ cudaFuncSetAttribute(sgemm_sm89_kernel<CfgM, true, IsSplitK, Layout>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size); cfg=true; }
        sgemm_sm89_kernel<CfgM, true, IsSplitK, Layout><<<dim3((N+127)/128, (M+127)/128, batchCount * splitK), 128, smem_size, handle->stream>>>(
            M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, beta, d_C, ldc, strideC, bias, bias_stride, batchCount, splitK);
    } else {
        constexpr size_t smem_size = CfgL::SMEM_BYTES;
        static bool cfg = false; if(!cfg){ cudaFuncSetAttribute(sgemm_sm89_kernel<CfgL, true, IsSplitK, Layout>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size); cfg=true; }
        sgemm_sm89_kernel<CfgL, true, IsSplitK, Layout><<<dim3((N+127)/128, (M+255)/256, batchCount * splitK), 256, smem_size, handle->stream>>>(
            M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, beta, d_C, ldc, strideC, bias, bias_stride, batchCount, splitK);
    }
}

template<SgemmLayout Layout>
void launch_sgemm_sm89(
    mycublasHandle_t handle, int M, int N, int K,
    float alpha, const float* d_A, int lda, long long int strideA,
    const float* d_B, int ldb, long long int strideB,
    float beta, float* d_C, int ldc, long long int strideC,
    const float* bias, long long bias_stride,
    int batchCount)
{
    launch_sgemm_sm89_templated<Layout, false>(handle, M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, beta, d_C, ldc, strideC, bias, bias_stride, batchCount, 1);
}

extern "C" {

void mycublasSgemmStridedBatched_nn_SM89(
    mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* d_A, int lda, long long int strideA,
    const float* d_B, int ldb, long long int strideB, const float beta, float* d_C, int ldc, long long int strideC, int batchCount)
{
    int sm_ver, sm_count;
    get_gpu_info(&sm_ver, &sm_count);
    if (sm_count <= 0) sm_count = 124; // Fallback

    // SplitK policy: routes backward grad-W matmuls (K=B*T large, M*N small →
    // total_blocks < sm_count) through K-striped CTAs with atomicAdd reduction.
    // Non-deterministic accumulation order biases weight updates by a few ULP;
    // over 19073 steps this costs ~5.7e-3 val loss vs cuBLAS deterministic
    // workspace reduction. Acceptable for throughput benchmarks; disable for
    // full parity runs by forcing splitK=1 below.
    int splitK = 1;
    {
        int blocks_per_batch = ((M + 255) / 256) * ((N + 127) / 128);
        int total_blocks = blocks_per_batch * batchCount;
        // Gate: trigger splitK when SM utilization < ~75% (total_blocks*4 < sm_count*3).
        // The tighter *2 threshold missed fc1/fc2 dW shapes (72 CTAs on 142 SMs = 50.7%).
        if (total_blocks * 4 < sm_count * 3 && K >= 512) {
            splitK = min(16, (sm_count + total_blocks - 1) / total_blocks);
        }
    }
    if (splitK > 1 && beta != 1.0f) { launch_sgemm_sm89_scale(d_C, beta, M, N, ldc, strideC, batchCount, handle->stream); }
    float final_beta = (splitK > 1) ? 1.0f : beta;

    if (M >= 256 && N >= 128) {
        launch_sgemm_nn_256x128_sm89_template(M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, final_beta, d_C, ldc, strideC, batchCount, splitK, handle->stream);
    } else if (M >= 128 && N >= 128) {
        launch_sgemm_nn_128x128_sm89_template(M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, final_beta, d_C, ldc, strideC, batchCount, splitK, handle->stream);
    } else if (M >= 64 && N >= 64) {
        launch_sgemm_nn_64x64_sm89_template(M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, final_beta, d_C, ldc, strideC, batchCount, splitK, handle->stream);
    } else if (M >= 32 && N >= 32) {
        launch_sgemm_nn_32x32_sm89_template(M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, final_beta, d_C, ldc, strideC, batchCount, splitK, handle->stream);
    } else {
        launch_sgemm_sm89_templated<SgemmLayout::NN, true>(handle, M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, beta, d_C, ldc, strideC, nullptr, 0, batchCount, splitK);
    }
}

void mycublasSgemmStridedBatched_nt_SM89(
    mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* d_A, int lda, long long int strideA,
    const float* d_B, int ldb, long long int strideB, const float beta, float* d_C, int ldc, long long int strideC, int batchCount)
{
    int sm_ver, sm_count;
    get_gpu_info(&sm_ver, &sm_count);
    if (sm_count <= 0) sm_count = 124;

    // SplitK policy: same rationale as NN dispatcher above. NT is the dominant
    // grad-W backward path (dW = X^T @ dY, K=B*T=16384, M=N=768 → 18 CTAs on
    // 142 SMs without splitK = 12.7% utilization). SplitK=8 → 144 CTAs ≈ full
    // SM saturation. Non-deterministic over full training; acceptable for
    // throughput benchmarks.
    int splitK = 1;
    {
        int blocks_per_batch = ((M + 255) / 256) * ((N + 127) / 128);
        int total_blocks = blocks_per_batch * batchCount;
        if (total_blocks * 4 < sm_count * 3 && K >= 512) {
            splitK = min(16, (sm_count + total_blocks - 1) / total_blocks);
        }
    }
    if (splitK > 1 && beta != 1.0f) { launch_sgemm_sm89_scale(d_C, beta, M, N, ldc, strideC, batchCount, handle->stream); }
    float final_beta = (splitK > 1) ? 1.0f : beta;

    if (M >= 256 && N >= 128) {
        launch_sgemm_nt_256x128_sm89_2(M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, final_beta, d_C, ldc, strideC, batchCount, splitK, handle->stream);
    } else if (M >= 128 && N >= 128) {
        launch_sgemm_nt_128x128_sm89_template(M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, final_beta, d_C, ldc, strideC, batchCount, splitK, handle->stream);
    } else if (M >= 64 && N >= 64) {
        launch_sgemm_nt_64x64_sm89_template(M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, final_beta, d_C, ldc, strideC, batchCount, splitK, handle->stream);
    } else if (M >= 32 && N >= 32) {
        launch_sgemm_nt_32x32_sm89_template(M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, final_beta, d_C, ldc, strideC, batchCount, splitK, handle->stream);
    } else {
        launch_sgemm_sm89_templated<SgemmLayout::NT, true>(handle, M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, beta, d_C, ldc, strideC, nullptr, 0, batchCount, splitK);
    }
}

void mycublasSgemmStridedBatched_tn_SM89(
    mycublasHandle_t handle, int M, int N, int K, const float alpha, const float* d_A, int lda, long long int strideA,
    const float* d_B, int ldb, long long int strideB, const float beta, float* d_C, int ldc, long long int strideC, int batchCount)
{
    int sm_ver, sm_count;
    get_gpu_info(&sm_ver, &sm_count);
    if (sm_count <= 0) sm_count = 124;

    // SplitK policy: same rationale as NN/NT dispatchers. TN backward dW
    // (M=N=768, K=16384) produces only 18 CTAs at 256×128 tile on 142 SMs.
    // SplitK restores full utilization. Non-deterministic over full training.
    int splitK = 1;
    {
        int blocks_per_batch = ((M + 255) / 256) * ((N + 127) / 128);
        int total_blocks = blocks_per_batch * batchCount;
        if (total_blocks * 4 < sm_count * 3 && K >= 512) {
            splitK = min(16, (sm_count + total_blocks - 1) / total_blocks);
        }
    }
    if (splitK > 1 && beta != 1.0f) { launch_sgemm_sm89_scale(d_C, beta, M, N, ldc, strideC, batchCount, handle->stream); }
    float final_beta = (splitK > 1) ? 1.0f : beta;

    if (M >= 256 && N >= 128) {
        launch_sgemm_tn_256x128_sm89_template(M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, final_beta, d_C, ldc, strideC, batchCount, splitK, handle->stream);
    } else if (M >= 128 && N >= 128) {
        launch_sgemm_tn_128x128_sm89_template(M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, final_beta, d_C, ldc, strideC, batchCount, splitK, handle->stream);
    } else if (M >= 64 && N >= 64) {
        launch_sgemm_tn_64x64_sm89_template(M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, final_beta, d_C, ldc, strideC, batchCount, splitK, handle->stream);
    } else if (M >= 32 && N >= 32) {
        launch_sgemm_tn_32x32_sm89_template(M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, final_beta, d_C, ldc, strideC, batchCount, splitK, handle->stream);
    } else {
        launch_sgemm_sm89_templated<SgemmLayout::TN, true>(handle, M, N, K, alpha, d_A, lda, strideA, d_B, ldb, strideB, beta, d_C, ldc, strideC, nullptr, 0, batchCount, splitK);
    }
}

} // extern "C"
