// test_bgemm_sm103_bf16_tn_cluster.cu
// Benchmark and accuracy test: BluBridge SM103a BF16 batched TN Cluster GEMM vs cuBLAS.
// Target: NVIDIA B300 (SM 10.3a) — tcgen05.mma.cta_group::2.kind::f16
//
// Usage:
//   ./build/test_bgemm_sm103_bf16_tn_cluster
//

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>

#include "../src/level3/batched_gemm/BF16/SM103/Bgemm_sm103_bf16.cuh"

// ─── error helpers ────────────────────────────────────────────────────────────

#define CHECK_CUDA(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t e = (call); \
    if (e != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)e, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); } \
} while(0)

// ─── data init ────────────────────────────────────────────────────────────────

static void fill_rand_bf16(std::vector<__nv_bfloat16>& v, unsigned seed)
{
    srand(seed);
    for (auto& x : v)
        x = __float2bfloat16((float(rand()) / RAND_MAX) * 2.f - 1.f);
}

// ─── accuracy check ───────────────────────────────────────────────────────────

static double max_rel_err(
    const std::vector<__nv_bfloat16>& ref,
    const std::vector<__nv_bfloat16>& out)
{
    double max_e = 0.0;
    for (size_t i = 0; i < ref.size(); ++i)
    {
        double r = __bfloat162float(ref[i]);
        double o = __bfloat162float(out[i]);
        double e = std::abs(r - o) / std::max(1e-5, std::abs(r));
        max_e = std::max(max_e, e);
    }
    return max_e;
}

// ─── benchmark harness ────────────────────────────────────────────────────────

struct Result { double blublas_tflops; double cublas_tflops; double rel_err; bool pass; };

static Result run_test(
    int M, int N, int K, int batchCount,
    float alpha, float beta,
    bool use_256_tile,
    cublasHandle_t cublas,
    cudaStream_t stream)
{
    const long long elA = (long long)batchCount * M * K;
    const long long elB = (long long)batchCount * K * N; 
    const long long elC = (long long)batchCount * M * N;

    std::vector<__nv_bfloat16> hA(elA), hB(elB), hC_ref(elC, __float2bfloat16(0.f));

    fill_rand_bf16(hA, 42);
    fill_rand_bf16(hB, 99);

    __nv_bfloat16 *dA, *dB, *dC_ref, *dC_blu;
    CHECK_CUDA(cudaMalloc(&dA,     elA * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&dB,     elB * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&dC_ref, elC * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&dC_blu, elC * sizeof(__nv_bfloat16)));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), elA * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), elB * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC_ref, 0, elC * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMemset(dC_blu, 0, elC * sizeof(__nv_bfloat16)));

    // ── cuBLAS reference (batched TN sgemm) ────────
    // C[M,N] = A^T[M,K] * B[K,N]
    // Under column-major: C^T = B^T * A
    // Since B is [K,N] row-major -> dB is [N,K] col-major. We use OP_N to get B^T [N,K].
    // Since A is [K,M] row-major -> dA is [M,K] col-major. We use OP_T to get A [K,M].
    {
        float fa = alpha, fb = beta;
        CHECK_CUBLAS(cublasGemmStridedBatchedEx(
            cublas,
            CUBLAS_OP_N, CUBLAS_OP_T,
            N, M, K,
            &fa,
            dB, CUDA_R_16BF, N, (long long)K * N,
            dA, CUDA_R_16BF, M, (long long)M * K,
            &fb,
            dC_ref, CUDA_R_16BF, N, (long long)M * N,
            batchCount,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    // Warmup BluBridge (5 iters)
    for (int w = 0; w < 5; ++w)
    {
        if (use_256_tile)
            mycublasBgemmSM103_bf16_tn_cluster_256x256x64(
                nullptr, M, N, K, alpha, dA, dB, beta, dC_blu, batchCount);
        else
            mycublasBgemmSM103_bf16_tn_cluster_128x128x64(
                nullptr, M, N, K, alpha, dA, dB, beta, dC_blu, batchCount);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaGetLastError());

    // Timed BluBridge (20 iters)
    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));
    CHECK_CUDA(cudaEventRecord(t0, stream));
    for (int i = 0; i < 20; ++i)
    {
        if (use_256_tile)
            mycublasBgemmSM103_bf16_tn_cluster_256x256x64(
                nullptr, M, N, K, alpha, dA, dB, beta, dC_blu, batchCount);
        else
            mycublasBgemmSM103_bf16_tn_cluster_128x128x64(
                nullptr, M, N, K, alpha, dA, dB, beta, dC_blu, batchCount);
    }
    CHECK_CUDA(cudaEventRecord(t1, stream));
    CHECK_CUDA(cudaEventSynchronize(t1));
    CHECK_CUDA(cudaGetLastError());
    float blu_ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&blu_ms, t0, t1));

    // Warmup + timed cuBLAS (20 iters)
    for (int w = 0; w < 5; ++w)
    {
        float fa = alpha, fb = beta;
        CHECK_CUBLAS(cublasGemmStridedBatchedEx(
            cublas, CUBLAS_OP_N, CUBLAS_OP_T, N, M, K,
            &fa, dB, CUDA_R_16BF, N, (long long)K*N,
                 dA, CUDA_R_16BF, M, (long long)M*K,
            &fb, dC_ref, CUDA_R_16BF, N, (long long)M*N,
            batchCount, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaEventRecord(t0, stream));
    for (int i = 0; i < 20; ++i)
    {
        float fa = alpha, fb = beta;
        CHECK_CUBLAS(cublasGemmStridedBatchedEx(
            cublas, CUBLAS_OP_N, CUBLAS_OP_T, N, M, K,
            &fa, dB, CUDA_R_16BF, N, (long long)K*N,
                 dA, CUDA_R_16BF, M, (long long)M*K,
            &fb, dC_ref, CUDA_R_16BF, N, (long long)M*N,
            batchCount, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CHECK_CUDA(cudaEventRecord(t1, stream));
    CHECK_CUDA(cudaEventSynchronize(t1));
    float cub_ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&cub_ms, t0, t1));

    // Accuracy
    std::vector<__nv_bfloat16> hC_ref_h(elC), hC_blu_h(elC);
    CHECK_CUDA(cudaMemcpy(hC_ref_h.data(), dC_ref, elC*sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hC_blu_h.data(), dC_blu, elC*sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    double rel_err = max_rel_err(hC_ref_h, hC_blu_h);

    double flops    = 2.0 * (double)M * N * K * batchCount;
    double iters    = 20.0;
    double blu_tfl  = flops * iters / (blu_ms * 1e-3) * 1e-12;
    double cub_tfl  = flops * iters / (cub_ms * 1e-3) * 1e-12;

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC_ref));
    CHECK_CUDA(cudaFree(dC_blu));
    CHECK_CUDA(cudaEventDestroy(t0));
    CHECK_CUDA(cudaEventDestroy(t1));

    return { blu_tfl, cub_tfl, rel_err, rel_err < 0.05 };
}

int main()
{
    int dev = 0;
    CHECK_CUDA(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

    printf("Device : %s  (SM %d.%d)\n", prop.name, prop.major, prop.minor);

    if (prop.major != 10 || prop.minor != 3)
    {
        printf("[SKIP] SM103 BF16 TN Cluster kernel requires SM 10.3 (B300/sm_103a).\n"
               "       This device is SM %d.%d (%s).\n"
               "       Tests skipped.\n",
               prop.major, prop.minor, prop.name);
        return EXIT_SUCCESS;
    }

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cublasHandle_t cublas;
    CHECK_CUBLAS(cublasCreate(&cublas));
    CHECK_CUBLAS(cublasSetStream(cublas, stream));

    struct Shape { int M, N, K, batch; const char* label; };
    
    // --- 128x128x64 Cluster Configurations ---
    const std::vector<Shape> shapes_128 = {
        {  128,  128,   64,  1, "tiny" },
        {  256,  256,  128,  1, "small" },
        {  512,  512,  128,  1, "medium" },
        { 1024, 1024,  128,  1, "large" },
        { 2048, 2048,  256,  1, "xlarge" },
        { 4096, 4096,  512,  1, "llm-style" },
        { 1024, 1024, 1024,  1, "square-1024" },
        { 2048, 2048, 2048,  1, "square-2048" },
        { 4096, 4096, 4096,  1, "square-4096" }
    };

    // --- 256x256x64 Cluster Configurations (Requires M/N multiples of 256) ---
    const std::vector<Shape> shapes_256 = {
        {  256,  256,  128,  1, "small-256" },
        {  512,  512,  128,  1, "medium-256" },
        { 1024, 1024,  128,  1, "large-256" },
        { 2048, 2048,  256,  1, "xlarge-256" },
        { 4096, 4096,  512,  1, "llm-style-256" },
        { 2048, 2048, 2048,  1, "square-2048-256" },
        { 4096, 4096, 4096,  1, "square-4096-256" }
    };

    printf("\n=== Testing 128x128x64 TN Cluster Kernel (cta_group::2) ===\n");
    printf("%-20s %6s %6s %6s %5s  %-8s  %-8s  %-10s  %s\n",
           "Shape", "M", "N", "K", "Batch",
           "BluBLAS", "cuBLAS", "MaxRelErr", "Status");
    printf("%s\n", std::string(85, '-').c_str());

    int passed = 0, total = (int)shapes_128.size();
    for (const auto& s : shapes_128)
    {
        Result r = run_test(s.M, s.N, s.K, s.batch,
                            1.0f, 0.0f, false, cublas, stream);
        printf("%-20s %6d %6d %6d %5d  %7.2f T  %7.2f T  %9.2e  %s\n",
               s.label, s.M, s.N, s.K, s.batch,
               r.blublas_tflops, r.cublas_tflops, r.rel_err,
               r.pass ? "PASS" : "FAIL");
        if (r.pass) ++passed;
    }
    printf("128x128x64 TN Results: %d / %d PASSED\n", passed, total);

    printf("\n=== Testing 256x256x64 TN Cluster Kernel (cta_group::2) ===\n");
    printf("%-20s %6s %6s %6s %5s  %-8s  %-8s  %-10s  %s\n",
           "Shape", "M", "N", "K", "Batch",
           "BluBLAS", "cuBLAS", "MaxRelErr", "Status");
    printf("%s\n", std::string(85, '-').c_str());

    int passed_256 = 0, total_256 = (int)shapes_256.size();
    for (const auto& s : shapes_256)
    {
        Result r = run_test(s.M, s.N, s.K, s.batch,
                            1.0f, 0.0f, true, cublas, stream);
        printf("%-20s %6d %6d %6d %5d  %7.2f T  %7.2f T  %9.2e  %s\n",
               s.label, s.M, s.N, s.K, s.batch,
               r.blublas_tflops, r.cublas_tflops, r.rel_err,
               r.pass ? "PASS" : "FAIL");
        if (r.pass) ++passed_256;
    }
    printf("256x256x64 TN Results: %d / %d PASSED\n\n", passed_256, total_256);

    CHECK_CUBLAS(cublasDestroy(cublas));
    CHECK_CUDA(cudaStreamDestroy(stream));

    return (passed == total && passed_256 == total_256) ? EXIT_SUCCESS : EXIT_FAILURE;
}
