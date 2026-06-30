// test_bgemm_sm103_bf16.cu
// Benchmark and accuracy test: BluBridge SM100a (cta_group::1) BF16 batched GEMM vs cuBLAS.
// Target: NVIDIA B100/B200 (SM 10.0a) — tcgen05.mma.cta_group::1.kind::f16
//
// NOTE: SM103a (B300 / SM 10.3) does NOT support tcgen05.mma.kind::f16.
// Per PTX ISA 9.3 p.757, B300 only supports tcgen05.mma.sp.kind::mxf4 and
// kind::mxf4nvf4 (MX-format sparse quantized MMA). Standard dense BF16 GEMM
// on B300 requires cuBLAS. This test skips on SM 10.3.
//
// Usage:
//   ./build/test_bgemm_sm103_bf16
//
// Output: per-shape TFLOPS comparison, max-relative-error, and pass/fail.

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

// Max relative error: |ref - out| / max(1e-5, |ref|)
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
    cublasHandle_t cublas,
    cudaStream_t stream)
{
    const long long elA = (long long)batchCount * M * K;
    const long long elB = (long long)batchCount * N * K;
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

    // ── cuBLAS reference (batched sgemm via cublasGemmStridedBatchedEx) ────────
    // cuBLAS uses column-major. For row-major A[M,K] × B[N,K]^T = C[M,N]:
    //   cublasGemmStridedBatchedEx(N, M, K) with B^T as "A" and A as "B"
    //   op(B) = BLAS_OP_N, op(A) = BLAS_OP_T → C = B × A^T in col-major
    //   but since C is col-major, that gives row-major result.
    {
        float fa = alpha, fb = beta;
        CHECK_CUBLAS(cublasGemmStridedBatchedEx(
            cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &fa,
            dB, CUDA_R_16BF, K, (long long)N * K,
            dA, CUDA_R_16BF, K, (long long)M * K,
            &fb,
            dC_ref, CUDA_R_16BF, N, (long long)M * N,
            batchCount,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    // Warmup BluBridge (5 iters)
    for (int w = 0; w < 5; ++w)
        mycublasBgemmSM103_bf16_nt_128x128x64(
            nullptr, M, N, K, alpha, dA, dB, beta, dC_blu, batchCount);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaGetLastError());

    // Timed BluBridge (20 iters)
    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));
    CHECK_CUDA(cudaEventRecord(t0, stream));
    for (int i = 0; i < 20; ++i)
        mycublasBgemmSM103_bf16_nt_128x128x64(
            nullptr, M, N, K, alpha, dA, dB, beta, dC_blu, batchCount);
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
            cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
            &fa, dB, CUDA_R_16BF, K, (long long)N*K,
                 dA, CUDA_R_16BF, K, (long long)M*K,
            &fb, dC_ref, CUDA_R_16BF, N, (long long)M*N,
            batchCount, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaEventRecord(t0, stream));
    for (int i = 0; i < 20; ++i)
    {
        float fa = alpha, fb = beta;
        CHECK_CUBLAS(cublasGemmStridedBatchedEx(
            cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
            &fa, dB, CUDA_R_16BF, K, (long long)N*K,
                 dA, CUDA_R_16BF, K, (long long)M*K,
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

// ─── main ─────────────────────────────────────────────────────────────────────

int main()
{
    // Check device — require SM103a (B300)
    int dev = 0;
    CHECK_CUDA(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

    printf("Device : %s  (SM %d.%d)\n", prop.name, prop.major, prop.minor);

    if (prop.major != 10 || prop.minor != 3)
    {
        printf("[SKIP] SM103 BF16 tcgen05 kernel requires SM 10.3 (B300/sm_103a).\n"
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

    // Test shapes: all must be multiples of 128×128×64
    struct Shape { int M, N, K, batch; const char* label; };
    const std::vector<Shape> shapes = {
        {  128,  128,   64,  1, "tiny" },
        {  256,  256,  128,  1, "small" },
        {  512,  512,  128,  1, "medium" },
        { 1024, 1024,  128,  1, "large" },
        { 2048, 2048,  256,  1, "xlarge" },
        {  128,  128,   64,  8, "batch-8-tiny" },
        {  256,  256,  128,  4, "batch-4-small" },
        {  512,  512,  128,  2, "batch-2-medium" },
        { 1024, 1024,  128,  2, "batch-2-large" },
        { 4096, 4096,  512,  1, "llm-style" },

        // Square shapes
        { 1024, 1024, 1024,  1, "square-1024" },
        { 2048, 2048, 2048,  1, "square-2048" },
        { 4096, 4096, 4096,  1, "square-4096" },
        { 8192, 8192, 8192,  1, "square-8192" },
        { 16384, 16384, 16384, 1, "square-16384" },

        // 120M Model Shapes (d_model=768, FFN=2048, Context=4096)
        { 4096, 1280,  768,  1, "120M-qkv" },
        { 4096,  768,  768,  1, "120M-o" },
        { 4096, 2048,  768,  1, "120M-ffn1" },
        { 4096,  768, 2048,  1, "120M-ffn2" },

        // 600M Model Shapes (d_model=1536, FFN=4096, Context=4096)
        { 4096, 2560, 1536,  1, "600M-qkv" },
        { 4096, 1536, 1536,  1, "600M-o" },
        { 4096, 4096, 1536,  1, "600M-ffn1" },
        { 4096, 1536, 4096,  1, "600M-ffn2" },

        // 1.7B Model Shapes (d_model=2048, FFN=5632, Context=4096)
        { 4096, 3072, 2048,  1, "1.7B-qkv" },
        { 4096, 2048, 2048,  1, "1.7B-o" },
        { 4096, 5632, 2048,  1, "1.7B-ffn1" },
        { 4096, 2048, 5632,  1, "1.7B-ffn2" },

        // 4B Model Shapes (d_model=3072, FFN=8192, Context=4096)
        { 4096, 5120, 3072,  1, "4B-qkv" },
        { 4096, 3072, 3072,  1, "4B-o" },
        { 4096, 8192, 3072,  1, "4B-ffn1" },
        { 4096, 3072, 8192,  1, "4B-ffn2" },

        // 7B Model Shapes (d_model=4096, FFN=14336, Context=4096)
        { 4096, 6144, 4096,  1, "7B-qkv" },
        { 4096, 4096, 4096,  1, "7B-o" },
        { 4096, 14336, 4096, 1, "7B-ffn1" },
        { 4096, 4096, 14336, 1, "7B-ffn2" },
    };

    printf("\n%-20s %6s %6s %6s %5s  %-8s  %-8s  %-10s  %s\n",
           "Shape", "M", "N", "K", "Batch",
           "BluBLAS", "cuBLAS", "MaxRelErr", "Status");
    printf("%s\n", std::string(85, '-').c_str());

    int passed = 0, total = (int)shapes.size();
    for (const auto& s : shapes)
    {
        Result r = run_test(s.M, s.N, s.K, s.batch,
                            1.0f, 0.0f, cublas, stream);
        printf("%-20s %6d %6d %6d %5d  %7.2f T  %7.2f T  %9.2e  %s\n",
               s.label, s.M, s.N, s.K, s.batch,
               r.blublas_tflops, r.cublas_tflops, r.rel_err,
               r.pass ? "PASS" : "FAIL");
        if (r.pass) ++passed;
    }

    printf("%s\n", std::string(85, '-').c_str());
    printf("Results: %d / %d PASSED\n\n", passed, total);

    CHECK_CUBLAS(cublasDestroy(cublas));
    CHECK_CUDA(cudaStreamDestroy(stream));

    return (passed == total) ? EXIT_SUCCESS : EXIT_FAILURE;
}
