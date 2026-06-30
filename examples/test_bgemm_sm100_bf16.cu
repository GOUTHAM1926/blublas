// test_bgemm_sm100_bf16.cu
// Performance and accuracy comparison: BluBridge SM100 BF16 NT kernel vs cuBLAS.
//
// Layout: C[b,M,N] = alpha * A[b,M,K] × B[b,N,K]^T + beta * C[b,M,N]
//   A : [batchCount, M, K]  row-major
//   B : [batchCount, N, K]  row-major   (B transposed = NT)
//   C : [batchCount, M, N]  row-major
//
// Build:
//   make test TEST=examples/test_bgemm_sm100_bf16.cu
// Run:
//   make run  TEST=test_bgemm_sm100_bf16

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

// ─── SM100 kernel declaration ─────────────────────────────────────────────────
// Declared here directly so the test doesn't depend on any extra header path.
extern "C" void mycublasBgemmSM100_bf16_nt_128x128x64(
    void*   handle,   // mycublasHandle_t — pass nullptr
    int M, int N, int K,
    float alpha,
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    float beta,
          __nv_bfloat16* C,
    int batchCount);

// ─── helpers ─────────────────────────────────────────────────────────────────

#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t _e = (call);                                           \
        if (_e != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(_e));           \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

#define CHECK_CUBLAS(call)                                                 \
    do {                                                                   \
        cublasStatus_t _s = (call);                                        \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "cuBLAS error %s:%d  code=%d\n",              \
                    __FILE__, __LINE__, (int)_s);                          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Simple xorshift PRNG for deterministic fill.
static uint32_t xorshift_state = 0xDEADBEEFu;
static inline float xorshift_float()
{
    xorshift_state ^= xorshift_state << 13;
    xorshift_state ^= xorshift_state >> 17;
    xorshift_state ^= xorshift_state <<  5;
    // Map to [-1, 1] with scale so the magnitude of dot products stays reasonable.
    return (float)(int)xorshift_state * (1.0f / (float)0x80000000u) * 0.5f;
}

static void fill_bf16(std::vector<__nv_bfloat16>& buf)
{
    for (auto& v : buf) v = __float2bfloat16(xorshift_float());
}

// Measure elapsed ms between two CUDA events.
static float event_ms(cudaEvent_t start, cudaEvent_t stop)
{
    float ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

// ─── accuracy metrics ─────────────────────────────────────────────────────────

struct Accuracy {
    float max_abs;   // max |ref - test|
    float max_rel;   // max |ref - test| / (|ref| + 1e-6)
    float rms_abs;   // sqrt(mean((ref-test)^2))
    bool  pass;      // max_abs < tol
};

static Accuracy compare_bf16(const std::vector<__nv_bfloat16>& ref,
                              const std::vector<__nv_bfloat16>& tst,
                              float abs_tol)
{
    Accuracy a{0.f, 0.f, 0.f, false};
    double sum_sq = 0.0;
    size_t n = ref.size();
    for (size_t i = 0; i < n; ++i) {
        float r = __bfloat162float(ref[i]);
        float t = __bfloat162float(tst[i]);
        float diff = fabsf(r - t);
        a.max_abs  = std::max(a.max_abs, diff);
        a.max_rel  = std::max(a.max_rel, diff / (fabsf(r) + 1e-6f));
        sum_sq    += (double)diff * diff;
    }
    a.rms_abs = (float)sqrt(sum_sq / (double)n);
    a.pass    = a.max_abs < abs_tol;
    return a;
}

// ─── benchmark harness ────────────────────────────────────────────────────────

struct BenchResult {
    float  ms_mean;    // mean kernel time (ms)
    double tflops;     // effective TFLOPS
};

static BenchResult benchmark_cublas(
    cublasHandle_t cublas,
    int M, int N, int K, int batch,
    const __nv_bfloat16* dA, const __nv_bfloat16* dB, __nv_bfloat16* dC,
    float alpha, float beta,
    int warmup, int iters)
{
    long long strideA = (long long)M * K;
    long long strideB = (long long)N * K;
    long long strideC = (long long)M * N;

    // cuBLAS operates column-major. For row-major NT (A[M,K] × B[N,K]^T = C[M,N]):
    //   col-major equivalent: C^T[N,M] = B[N,K] × A[K,M]^T
    //   → cublasGemmStridedBatchedEx(OP_N, OP_T, N, M, K, B, N, A, K, C, N)
    int lda = K, ldb = K, ldc = N;

    float h_alpha = alpha, h_beta = beta;

    auto run = [&]() {
        CHECK_CUBLAS(cublasGemmStridedBatchedEx(
            cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &h_alpha,
            dB, CUDA_R_16BF, ldb, strideB,
            dA, CUDA_R_16BF, lda, strideA,
            &h_beta,
            dC, CUDA_R_16BF, ldc, strideC,
            batch,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };

    for (int i = 0; i < warmup; ++i) run();
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t ev0, ev1;
    CHECK_CUDA(cudaEventCreate(&ev0));
    CHECK_CUDA(cudaEventCreate(&ev1));
    CHECK_CUDA(cudaEventRecord(ev0));
    for (int i = 0; i < iters; ++i) run();
    CHECK_CUDA(cudaEventRecord(ev1));
    CHECK_CUDA(cudaEventSynchronize(ev1));

    float total_ms = event_ms(ev0, ev1);
    float mean_ms  = total_ms / iters;
    double ops     = 2.0 * M * N * K * batch;
    double tflops  = ops / (mean_ms * 1e-3) / 1e12;

    CHECK_CUDA(cudaEventDestroy(ev0));
    CHECK_CUDA(cudaEventDestroy(ev1));
    return {mean_ms, tflops};
}

static BenchResult benchmark_sm100(
    int M, int N, int K, int batch,
    const __nv_bfloat16* dA, const __nv_bfloat16* dB, __nv_bfloat16* dC,
    float alpha, float beta,
    int warmup, int iters)
{
    auto run = [&]() {
        mycublasBgemmSM100_bf16_nt_128x128x64(
            nullptr, M, N, K, alpha, dA, dB, beta, dC, batch);
        cudaError_t kerr = cudaGetLastError();
        if (kerr != cudaSuccess)
            fprintf(stderr, "  [SM100 bench error: %s]\n",
                    cudaGetErrorString(kerr));
    };

    for (int i = 0; i < warmup; ++i) run();
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t ev0, ev1;
    CHECK_CUDA(cudaEventCreate(&ev0));
    CHECK_CUDA(cudaEventCreate(&ev1));
    CHECK_CUDA(cudaEventRecord(ev0));
    for (int i = 0; i < iters; ++i) run();
    CHECK_CUDA(cudaEventRecord(ev1));
    CHECK_CUDA(cudaEventSynchronize(ev1));

    float total_ms = event_ms(ev0, ev1);
    float mean_ms  = total_ms / iters;
    double ops     = 2.0 * M * N * K * batch;
    double tflops  = ops / (mean_ms * 1e-3) / 1e12;

    CHECK_CUDA(cudaEventDestroy(ev0));
    CHECK_CUDA(cudaEventDestroy(ev1));
    return {mean_ms, tflops};
}

// ─── single test case ─────────────────────────────────────────────────────────

struct TestCase {
    int M, N, K, batch;
    std::string label;
};

static bool run_test_case(
    cublasHandle_t cublas,
    const TestCase& tc,
    float alpha, float beta,
    int warmup, int iters,
    float abs_tol)
{
    const int M = tc.M, N = tc.N, K = tc.K, B = tc.batch;

    // Alignment check: kernel requires M%128==0, N%128==0, K%64==0
    if (M % 128 != 0 || N % 128 != 0 || K % 64 != 0) {
        printf("  [SKIP] %s  M=%d N=%d K=%d B=%d  (alignment)\n",
               tc.label.c_str(), M, N, K, B);
        return true;
    }

    size_t szA = (size_t)B * M * K;
    size_t szB = (size_t)B * N * K;
    size_t szC = (size_t)B * M * N;

    // Host buffers
    std::vector<__nv_bfloat16> hA(szA), hB(szB), hC_ref(szC), hC_sm100(szC);
    fill_bf16(hA);
    fill_bf16(hB);

    // Device buffers
    __nv_bfloat16 *dA, *dB, *dC_ref, *dC_sm100;
    CHECK_CUDA(cudaMalloc(&dA,      szA * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&dB,      szB * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&dC_ref,  szC * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&dC_sm100,szC * sizeof(__nv_bfloat16)));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), szA * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), szB * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    // Zero C outputs
    CHECK_CUDA(cudaMemset(dC_ref,   0, szC * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMemset(dC_sm100, 0, szC * sizeof(__nv_bfloat16)));

    // ── accuracy pass (single run) ────────────────────────────────────────────
    {
        long long sA = (long long)M * K, sB = (long long)N * K, sC = (long long)M * N;
        float h_alpha = alpha, h_beta = beta;

        // cuBLAS reference (col-major equivalent of row-major NT)
        CHECK_CUBLAS(cublasGemmStridedBatchedEx(
            cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &h_alpha,
            dB, CUDA_R_16BF, K, sB,
            dA, CUDA_R_16BF, K, sA,
            &h_beta,
            dC_ref, CUDA_R_16BF, N, sC,
            B, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        // SM100 kernel
        mycublasBgemmSM100_bf16_nt_128x128x64(
            nullptr, M, N, K, alpha, dA, dB, beta, dC_sm100, B);
        {
            cudaError_t kerr = cudaGetLastError();
            if (kerr != cudaSuccess)
                fprintf(stderr, "  [SM100 kernel launch error: %s]\n",
                        cudaGetErrorString(kerr));
        }
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    CHECK_CUDA(cudaMemcpy(hC_ref.data(),   dC_ref,   szC * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hC_sm100.data(), dC_sm100, szC * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    Accuracy acc = compare_bf16(hC_ref, hC_sm100, abs_tol);

    // ── benchmark ─────────────────────────────────────────────────────────────
    BenchResult res_cublas = benchmark_cublas(cublas, M, N, K, B, dA, dB, dC_ref,
                                              alpha, beta, warmup, iters);
    BenchResult res_sm100  = benchmark_sm100 (M, N, K, B, dA, dB, dC_sm100,
                                              alpha, beta, warmup, iters);

    float speedup = (float)(res_sm100.tflops / res_cublas.tflops);

    // ── print ─────────────────────────────────────────────────────────────────
    printf("  %-28s  M=%4d N=%4d K=%4d B=%2d\n",
           tc.label.c_str(), M, N, K, B);
    printf("    cuBLAS : %7.3f ms  %7.2f TFLOPS\n",
           res_cublas.ms_mean, res_cublas.tflops);
    printf("    SM100  : %7.3f ms  %7.2f TFLOPS  (%.2fx)\n",
           res_sm100.ms_mean, res_sm100.tflops, speedup);
    printf("    Accuracy: max_abs=%.4f  max_rel=%.4f%%  rms=%.6f  [%s]\n",
           acc.max_abs, acc.max_rel * 100.f, acc.rms_abs,
           acc.pass ? "PASS" : "FAIL");
    printf("\n");

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC_ref));
    CHECK_CUDA(cudaFree(dC_sm100));

    return acc.pass;
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main()
{
    // Device info
    int dev = 0;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s  (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    // SM100 kernel requires exactly SM 10.0 (GB100/B100/B200/GB200).
    // SM 10.3 (B300) does NOT support tcgen05.mma per PTX ISA 9.3.
    if (prop.major != 10 || prop.minor != 0) {
        printf("[SKIP] SM100 BF16 tcgen05 kernel requires SM 10.0 (B100/B200/GB200).\n"
               "       This device is SM %d.%d (%s).\n"
               "       tcgen05.mma / .alloc / .ld.32x32b are NOT in sm_103f/sm_103a\n"
               "       per PTX ISA 9.3 Table 70. Tests skipped.\n",
               prop.major, prop.minor, prop.name);
        return EXIT_SUCCESS;
    }

    cublasHandle_t cublas;
    CHECK_CUBLAS(cublasCreate(&cublas));

    // ── test matrix ──────────────────────────────────────────────────────────
    // BF16 accumulates with FP32 internally; cuBLAS and SM100 kernel both use
    // TMEM/FP32 accumulators. Tolerance of 0.5 handles rounding-order differences.
    const float ABS_TOL = 0.5f;
    const float alpha   = 1.0f;
    const float beta    = 0.0f;
    const int   WARMUP  = 5;
    const int   ITERS   = 20;

    std::vector<TestCase> cases = {
        // Single tile (minimum aligned problem)
        {128,  128,   64, 1,  "single tile"},
        // Small square
        {512,  512,  512, 1,  "sq-512"},
        // Transformer attention head (1 batch)
        {1024, 1024,  64, 1,  "attn-head (s=1024,d=64)"},
        // Batched attention: 12 heads, seq=512, head_dim=64
        {512,  512,   64, 12, "attn-batched (B=12,s=512)"},
        // LLM-scale: batch=32 attention
        {1024, 1024,  64, 32, "attn-large  (B=32,s=1024)"},
        // Medium square
        {1024, 1024, 1024, 1, "sq-1024"},
        // Large square
        {2048, 2048, 2048, 1, "sq-2048"},
        // Projection layers (non-square)
        {2048, 512,  2048, 4, "proj-wide  (B=4)"},
        {512,  2048, 2048, 4, "proj-tall  (B=4)"},
        // Very large
        {4096, 4096, 4096, 1, "sq-4096"},
    };

    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  BluBridge SM100 BF16 NT  vs  cuBLAS  —  Performance + Accuracy ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    printf("alpha=%.1f  beta=%.1f  warmup=%d  iters=%d  abs_tol=%.2f\n\n",
           alpha, beta, WARMUP, ITERS, ABS_TOL);

    int passed = 0, total = 0;
    for (const auto& tc : cases) {
        bool ok = run_test_case(cublas, tc, alpha, beta, WARMUP, ITERS, ABS_TOL);
        if (ok) ++passed;
        ++total;
    }

    printf("─────────────────────────────────────────────────────────────────\n");
    printf("Accuracy summary: %d / %d PASSED\n", passed, total);
    printf("─────────────────────────────────────────────────────────────────\n\n");

    // ── beta != 0 sanity check ────────────────────────────────────────────────
    printf("Beta blending test (alpha=1.0, beta=0.5) ...\n");
    {
        int M = 512, N = 512, K = 128, B = 1;
        size_t szA = M * K, szB = N * K, szC = M * N;

        std::vector<__nv_bfloat16> hA(szA), hB(szB), hC_init(szC);
        fill_bf16(hA); fill_bf16(hB); fill_bf16(hC_init);

        __nv_bfloat16 *dA, *dB, *dC_ref, *dC_sm100;
        CHECK_CUDA(cudaMalloc(&dA,       szA * sizeof(__nv_bfloat16)));
        CHECK_CUDA(cudaMalloc(&dB,       szB * sizeof(__nv_bfloat16)));
        CHECK_CUDA(cudaMalloc(&dC_ref,   szC * sizeof(__nv_bfloat16)));
        CHECK_CUDA(cudaMalloc(&dC_sm100, szC * sizeof(__nv_bfloat16)));

        CHECK_CUDA(cudaMemcpy(dA, hA.data(), szA * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dB, hB.data(), szB * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        // Pre-fill C with the same initial values for both runs.
        CHECK_CUDA(cudaMemcpy(dC_ref,   hC_init.data(), szC * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dC_sm100, hC_init.data(), szC * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

        float a = 1.0f, b = 0.5f;

        CHECK_CUBLAS(cublasGemmStridedBatchedEx(
            cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &a,
            dB, CUDA_R_16BF, K, (long long)N*K,
            dA, CUDA_R_16BF, K, (long long)M*K,
            &b,
            dC_ref, CUDA_R_16BF, N, (long long)M*N,
            B, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        mycublasBgemmSM100_bf16_nt_128x128x64(
            nullptr, M, N, K, a, dA, dB, b, dC_sm100, B);

        CHECK_CUDA(cudaDeviceSynchronize());

        std::vector<__nv_bfloat16> hC_ref(szC), hC_sm100(szC);
        CHECK_CUDA(cudaMemcpy(hC_ref.data(),   dC_ref,   szC * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(hC_sm100.data(), dC_sm100, szC * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

        Accuracy acc = compare_bf16(hC_ref, hC_sm100, ABS_TOL);
        printf("  Beta=0.5 test: max_abs=%.4f  rms=%.6f  [%s]\n\n",
               acc.max_abs, acc.rms_abs, acc.pass ? "PASS" : "FAIL");

        CHECK_CUDA(cudaFree(dA));
        CHECK_CUDA(cudaFree(dB));
        CHECK_CUDA(cudaFree(dC_ref));
        CHECK_CUDA(cudaFree(dC_sm100));
    }

    CHECK_CUBLAS(cublasDestroy(cublas));
    printf("Done.\n");
    return (passed == total) ? EXIT_SUCCESS : EXIT_FAILURE;
}
