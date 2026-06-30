// test_bgemm_sm103_cluster_template.cu
// Comprehensive accuracy and benchmark test suite for the unified SM103a Cluster GEMM template.
// Target: NVIDIA B300 (SM 10.3a) — tcgen05.mma.cta_group::2.kind::f16
//
// Usage:
//   make run TEST=examples/test_bgemm_sm103_cluster_template.cu
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

// Include the unified SM103 cluster template
#include "../src/level3/batched_gemm/BF16/SM103/Bgemm_sm103_cluster_template.cuh"

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

struct Result {
    double blublas_tflops;
    double cublas_tflops;
    double rel_err;
    bool pass;
};

// Generic runner for any Config and Layout
template<typename Config, Sm103Layout Layout>
static Result run_test_case(
    int M, int N, int K, int batchCount,
    float alpha, float beta,
    cublasHandle_t cublas,
    cudaStream_t stream)
{
    // Determine dimensions based on layout
    long long elA = 0;
    long long elB = 0;
    const long long elC = (long long)batchCount * M * N;

    if (Layout == Sm103Layout::NN) {
        elA = (long long)batchCount * M * K;
        elB = (long long)batchCount * K * N;
    } else if (Layout == Sm103Layout::NT) {
        elA = (long long)batchCount * M * K;
        elB = (long long)batchCount * N * K;
    } else { // TN
        elA = (long long)batchCount * K * M;
        elB = (long long)batchCount * K * N;
    }

    std::vector<__nv_bfloat16> hA(elA), hB(elB), hC_init(elC);
    fill_rand_bf16(hA, 42);
    fill_rand_bf16(hB, 99);
    
    // For beta != 0, initialize C with random values
    if (beta != 0.0f) {
        fill_rand_bf16(hC_init, 77);
    } else {
        std::fill(hC_init.begin(), hC_init.end(), __float2bfloat16(0.f));
    }

    __nv_bfloat16 *dA, *dB, *dC_ref, *dC_blu;
    CHECK_CUDA(cudaMalloc(&dA,     elA * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&dB,     elB * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&dC_ref, elC * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&dC_blu, elC * sizeof(__nv_bfloat16)));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), elA * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), elB * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC_ref, hC_init.data(), elC * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC_blu, hC_init.data(), elC * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    // ── cuBLAS reference ────────
    {
        float fa = alpha, fb = beta;
        cublasOperation_t transA, transB;
        long long lda, ldb, strideA, strideB;

        if (Layout == Sm103Layout::NN) {
            transA = CUBLAS_OP_N; transB = CUBLAS_OP_N;
            lda = K; ldb = N;
            strideA = (long long)M * K; strideB = (long long)K * N;
        } else if (Layout == Sm103Layout::NT) {
            transA = CUBLAS_OP_N; transB = CUBLAS_OP_T;
            lda = K; ldb = K;
            strideA = (long long)M * K; strideB = (long long)N * K;
        } else { // TN
            transA = CUBLAS_OP_T; transB = CUBLAS_OP_N;
            lda = M; ldb = N;
            strideA = (long long)K * M; strideB = (long long)K * N;
        }

        CHECK_CUBLAS(cublasGemmStridedBatchedEx(
            cublas,
            transB, transA, // Swap due to column-major vs row-major
            N, M, K,
            &fa,
            dB, CUDA_R_16BF, (int)ldb, strideB,
            dA, CUDA_R_16BF, (int)lda, strideA,
            &fb,
            dC_ref, CUDA_R_16BF, N, (long long)M * N,
            batchCount,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    // Run exactly one iteration of BluBridge for accuracy checking
    launch_bgemm_sm103_cluster<Config, Layout>(
        dA, dB, dC_blu, M, N, K, batchCount, alpha, beta, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaGetLastError());

    // Accuracy Check
    std::vector<__nv_bfloat16> hC_ref_h(elC), hC_blu_h(elC);
    CHECK_CUDA(cudaMemcpy(hC_ref_h.data(), dC_ref, elC*sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hC_blu_h.data(), dC_blu, elC*sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    double rel_err = max_rel_err(hC_ref_h, hC_blu_h);
    if (rel_err >= 0.05) {
        printf("  [DEBUG FAIL] First 10 elements:\n");
        printf("    Ref (cuBLAS): ");
        for (int i = 0; i < std::min(10, (int)elC); ++i) printf("%.4f ", __bfloat162float(hC_ref_h[i]));
        printf("\n    Blu (Template): ");
        for (int i = 0; i < std::min(10, (int)elC); ++i) printf("%.4f ", __bfloat162float(hC_blu_h[i]));
        printf("\n");
    }

    // Warmup BluBridge Cluster Template (5 iters)
    for (int w = 0; w < 5; ++w)
    {
        launch_bgemm_sm103_cluster<Config, Layout>(
            dA, dB, dC_blu, M, N, K, batchCount, alpha, beta, stream);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaGetLastError());

    // Timed BluBridge Cluster Template (20 iters)
    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));
    CHECK_CUDA(cudaEventRecord(t0, stream));
    for (int i = 0; i < 20; ++i)
    {
        launch_bgemm_sm103_cluster<Config, Layout>(
            dA, dB, dC_blu, M, N, K, batchCount, alpha, beta, stream);
    }
    CHECK_CUDA(cudaEventRecord(t1, stream));
    CHECK_CUDA(cudaEventSynchronize(t1));
    CHECK_CUDA(cudaGetLastError());
    float blu_ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&blu_ms, t0, t1));

    // Warmup + timed cuBLAS (20 iters)
    {
        float fa = alpha, fb = beta;
        cublasOperation_t transA, transB;
        long long lda, ldb, strideA, strideB;

        if (Layout == Sm103Layout::NN) {
            transA = CUBLAS_OP_N; transB = CUBLAS_OP_N;
            lda = K; ldb = N;
            strideA = (long long)M * K; strideB = (long long)K * N;
        } else if (Layout == Sm103Layout::NT) {
            transA = CUBLAS_OP_N; transB = CUBLAS_OP_T;
            lda = K; ldb = K;
            strideA = (long long)M * K; strideB = (long long)N * K;
        } else { // TN
            transA = CUBLAS_OP_T; transB = CUBLAS_OP_N;
            lda = M; ldb = N;
            strideA = (long long)K * M; strideB = (long long)K * N;
        }

        for (int w = 0; w < 5; ++w)
        {
            CHECK_CUBLAS(cublasGemmStridedBatchedEx(
                cublas, transB, transA, N, M, K, &fa,
                dB, CUDA_R_16BF, (int)ldb, strideB,
                dA, CUDA_R_16BF, (int)lda, strideA,
                &fb, dC_ref, CUDA_R_16BF, N, (long long)M * N,
                batchCount, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        CHECK_CUDA(cudaStreamSynchronize(stream));
        CHECK_CUDA(cudaEventRecord(t0, stream));
        for (int i = 0; i < 20; ++i)
        {
            CHECK_CUBLAS(cublasGemmStridedBatchedEx(
                cublas, transB, transA, N, M, K, &fa,
                dB, CUDA_R_16BF, (int)ldb, strideB,
                dA, CUDA_R_16BF, (int)lda, strideA,
                &fb, dC_ref, CUDA_R_16BF, N, (long long)M * N,
                batchCount, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        CHECK_CUDA(cudaEventRecord(t1, stream));
        CHECK_CUDA(cudaEventSynchronize(t1));
    }
    float cub_ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&cub_ms, t0, t1));

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

// Struct to define dynamic test shape
struct Shape {
    int M, N, K, batch;
    float alpha, beta;
    const char* label;
};

// Layout name helper
static const char* layout_name(Sm103Layout layout)
{
    switch (layout) {
        case Sm103Layout::NN: return "NN";
        case Sm103Layout::NT: return "NT";
        case Sm103Layout::TN: return "TN";
    }
    return "Unknown";
}

template<typename Config, Sm103Layout Layout>
static bool run_test_suite_for_layout_config(
    const std::vector<Shape>& shapes,
    cublasHandle_t cublas,
    cudaStream_t stream)
{
    printf("\n=== Testing Layout: %s | Config: %dx%dx%d (cta_group::2) ===\n",
           layout_name(Layout), Config::BM, Config::BN, Config::BK);
    printf("%-18s %5s %5s %5s %5s %5s %5s  %-9s  %-9s  %-10s  %s\n",
           "Shape", "M", "N", "K", "Batch", "alpha", "beta",
           "Template", "cuBLAS", "MaxRelErr", "Status");
    printf("%s\n", std::string(98, '-').c_str());

    int passed = 0;
    int total = 0;
    for (const auto& s : shapes)
    {
        // 256x256x64 config requires M & N to be multiples of 256
        if (Config::BM == 256 && (s.M % 512 != 0 || s.N % 256 != 0)) {
            // Skip shapes that are not compatible with 256x256 config
            // Note: cluster height is 2*BM = 512, cluster width is BN = 256
            continue;
        }

        total++;
        Result r = run_test_case<Config, Layout>(
            s.M, s.N, s.K, s.batch, s.alpha, s.beta, cublas, stream);
        
        printf("%-18s %5d %5d %5d %5d %5.1f %5.1f  %8.2f T  %8.2f T  %9.2e  %s\n",
               s.label, s.M, s.N, s.K, s.batch, s.alpha, s.beta,
               r.blublas_tflops, r.cublas_tflops, r.rel_err,
               r.pass ? "PASS" : "FAIL");
        
        if (r.pass) ++passed;
    }
    printf("Result: %d / %d PASSED\n", passed, total);
    return passed == total;
}

int main()
{
    int dev = 0;
    CHECK_CUDA(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

    printf("=========================================================================\n");
    printf("SM103 Cluster GEMM Unified Template Test Suite\n");
    printf("Device : %s  (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("=========================================================================\n");

    if (prop.major != 10 || prop.minor != 3)
    {
        printf("[SKIP] SM103 Cluster GEMM template requires SM 10.3 (Blackwell Ultra/sm_103a).\n"
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

    // Test cases for 128x128x64 configuration (requires M % 256 == 0, N % 128 == 0, K % 64 == 0)
    // Note: for BM=128, cluster height is 2*BM = 256. So M must be a multiple of 256.
    const std::vector<Shape> shapes_128 = {
        // M, N, K, batch, alpha, beta, label
        {  256,  128,   64,  1, 1.0f, 0.0f, "tiny" },
        {  256,  256,  128,  1, 1.0f, 0.0f, "small" },
        {  512,  512,  128,  1, 1.0f, 0.0f, "medium" },
        { 1024, 1024,  256,  1, 1.0f, 0.0f, "large" },
        // Multi-batch test
        {  512,  512,  128,  4, 1.0f, 0.0f, "batched-4" },
        {  256,  256,  128,  8, 1.0f, 0.0f, "batched-8" },
        // Alpha/Beta scaling and accumulation tests
        {  512,  512,  128,  1, 1.5f, 0.0f, "alpha-scaling" },
        {  256,  256,  128,  1, 1.0f, 0.5f, "beta-accumulate" },
        {  512,  512,  128,  2, 1.2f, 0.8f, "alpha-beta-batch" }
    };

    // Test cases for 256x256x64 configuration (requires M % 512 == 0, N % 256 == 0, K % 64 == 0)
    const std::vector<Shape> shapes_256 = {
        {  512,  256,  128,  1, 1.0f, 0.0f, "tiny-256" },
        {  512,  512,  128,  1, 1.0f, 0.0f, "small-256" },
        { 1024, 1024,  256,  1, 1.0f, 0.0f, "medium-256" },
        // Multi-batch test
        {  512,  512,  128,  4, 1.0f, 0.0f, "batched-4-256" },
        // Alpha/Beta scaling and accumulation tests
        {  512,  512,  128,  1, 1.5f, 0.5f, "alpha-beta-256" }
    };

    bool all_passed = true;

    // 1. NN Layout
    all_passed &= run_test_suite_for_layout_config<Sm103Config_128x128x64, Sm103Layout::NN>(shapes_128, cublas, stream);
    all_passed &= run_test_suite_for_layout_config<Sm103Config_256x256x64, Sm103Layout::NN>(shapes_256, cublas, stream);

    // 2. NT Layout
    all_passed &= run_test_suite_for_layout_config<Sm103Config_128x128x64, Sm103Layout::NT>(shapes_128, cublas, stream);
    all_passed &= run_test_suite_for_layout_config<Sm103Config_256x256x64, Sm103Layout::NT>(shapes_256, cublas, stream);

    // 3. TN Layout
    all_passed &= run_test_suite_for_layout_config<Sm103Config_128x128x64, Sm103Layout::TN>(shapes_128, cublas, stream);
    all_passed &= run_test_suite_for_layout_config<Sm103Config_256x256x64, Sm103Layout::TN>(shapes_256, cublas, stream);

    printf("\n=========================================================================\n");
    if (all_passed) {
        printf("ALL TESTS PASSED SUCCESSFULLY!\n");
    } else {
        printf("SOME TESTS FAILED!\n");
    }
    printf("=========================================================================\n");

    CHECK_CUBLAS(cublasDestroy(cublas));
    CHECK_CUDA(cudaStreamDestroy(stream));

    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
