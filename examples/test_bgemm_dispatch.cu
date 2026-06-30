// test_bgemm_dispatch.cu
// Unified accuracy + benchmark test for mycublasBgemmStridedBatched dispatcher.
// Tests NN / NT / TN layouts across a range of shapes.
// On SM103 (B300) the SM103 cluster kernels are invoked automatically.
// Falls back to SM89 / SM80 kernels on older hardware.
//
// Usage:
//   make run TEST=test_bgemm_dispatch

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

#include "../include/mycublas.h"

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

static void fill_rand_bf16(std::vector<__nv_bfloat16>& v, unsigned seed)
{
    srand(seed);
    for (auto& x : v)
        x = __float2bfloat16((float(rand()) / RAND_MAX) * 2.f - 1.f);
}

static double max_rel_err(
    const std::vector<__nv_bfloat16>& ref,
    const std::vector<__nv_bfloat16>& out)
{
    double max_e = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double r = __bfloat162float(ref[i]);
        double o = __bfloat162float(out[i]);
        double e = std::abs(r - o) / std::max(1e-5, std::abs(r));
        max_e = std::max(max_e, e);
    }
    return max_e;
}

enum Layout { NN, NT, TN };

struct Result { double blublas_tflops; double cublas_tflops; double rel_err; bool pass; };

static Result run_test(
    Layout layout,
    int M, int N, int K, int batchCount,
    float alpha_f, float beta_f,
    mycublasHandle_t myblas,
    cublasHandle_t cublas,
    cudaStream_t stream)
{
    // A dims: NN/NT=[M,K], TN=[K,M]
    // B dims: NN/TN=[K,N], NT=[N,K]
    long long elA = (layout == TN) ? (long long)batchCount * K * M
                                   : (long long)batchCount * M * K;
    long long elB = (layout == NT) ? (long long)batchCount * N * K
                                   : (long long)batchCount * K * N;
    long long elC = (long long)batchCount * M * N;

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

    float fa = alpha_f, fb = beta_f;

    // cuBLAS reference — all in column-major terms (C^T = op(B)^T * op(A)^T)
    if (layout == NN) {
        // C[M,N] = A[M,K] * B[K,N]
        // cuBLAS col-major: opA=N,opB=N  m=N,n=M,k=K  B leading=N, A leading=K
        CHECK_CUBLAS(cublasGemmStridedBatchedEx(
            cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
            &fa,
            dB, CUDA_R_16BF, N, (long long)K * N,
            dA, CUDA_R_16BF, K, (long long)M * K,
            &fb,
            dC_ref, CUDA_R_16BF, N, (long long)M * N,
            batchCount, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else if (layout == NT) {
        // C[M,N] = A[M,K] * B[N,K]^T
        // cuBLAS col-major: opA=T,opB=N  m=N,n=M,k=K  B leading=K, A leading=K
        CHECK_CUBLAS(cublasGemmStridedBatchedEx(
            cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
            &fa,
            dB, CUDA_R_16BF, K, (long long)N * K,
            dA, CUDA_R_16BF, K, (long long)M * K,
            &fb,
            dC_ref, CUDA_R_16BF, N, (long long)M * N,
            batchCount, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else { // TN
        // C[M,N] = A[K,M]^T * B[K,N]
        // cuBLAS col-major: opA=N,opB=T  m=N,n=M,k=K  B leading=N, A leading=M
        CHECK_CUBLAS(cublasGemmStridedBatchedEx(
            cublas, CUBLAS_OP_N, CUBLAS_OP_T, N, M, K,
            &fa,
            dB, CUDA_R_16BF, N, (long long)K * N,
            dA, CUDA_R_16BF, M, (long long)M * K,
            &fb,
            dC_ref, CUDA_R_16BF, N, (long long)M * N,
            batchCount, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Leading dims and strides for mycublasBgemmStridedBatched
    int ldA  = (layout == TN) ? M : K;
    int ldB  = (layout == NT) ? K : N;
    int ldC  = N;
    long long sA = (layout == TN) ? (long long)K * M : (long long)M * K;
    long long sB = (layout == NT) ? (long long)N * K : (long long)K * N;
    long long sC = (long long)M * N;

    mycublasOperation_t opA = (layout == TN) ? MYCUBLAS_OP_T : MYCUBLAS_OP_N;
    mycublasOperation_t opB = (layout == NT) ? MYCUBLAS_OP_T : MYCUBLAS_OP_N;

    __nv_bfloat16 alpha_h = __float2bfloat16(alpha_f);
    __nv_bfloat16 beta_h  = __float2bfloat16(beta_f);

    // Warmup
    for (int w = 0; w < 3; ++w)
        mycublasBgemmStridedBatched(myblas, opA, opB, M, N, K,
            alpha_h, dA, ldA, sA, dB, ldB, sB, beta_h, dC_blu, ldC, sC, batchCount);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaGetLastError());

    // Timed BluBridge
    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));
    CHECK_CUDA(cudaEventRecord(t0, stream));
    for (int i = 0; i < 20; ++i)
        mycublasBgemmStridedBatched(myblas, opA, opB, M, N, K,
            alpha_h, dA, ldA, sA, dB, ldB, sB, beta_h, dC_blu, ldC, sC, batchCount);
    CHECK_CUDA(cudaEventRecord(t1, stream));
    CHECK_CUDA(cudaEventSynchronize(t1));
    CHECK_CUDA(cudaGetLastError());
    float blu_ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&blu_ms, t0, t1));

    // Timed cuBLAS
    for (int w = 0; w < 3; ++w) {
        if (layout == NN)
            cublasGemmStridedBatchedEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                &fa, dB, CUDA_R_16BF, N, (long long)K*N,
                     dA, CUDA_R_16BF, K, (long long)M*K,
                &fb, dC_ref, CUDA_R_16BF, N, (long long)M*N,
                batchCount, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        else if (layout == NT)
            cublasGemmStridedBatchedEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                &fa, dB, CUDA_R_16BF, K, (long long)N*K,
                     dA, CUDA_R_16BF, K, (long long)M*K,
                &fb, dC_ref, CUDA_R_16BF, N, (long long)M*N,
                batchCount, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        else
            cublasGemmStridedBatchedEx(cublas, CUBLAS_OP_N, CUBLAS_OP_T, N, M, K,
                &fa, dB, CUDA_R_16BF, N, (long long)K*N,
                     dA, CUDA_R_16BF, M, (long long)M*K,
                &fb, dC_ref, CUDA_R_16BF, N, (long long)M*N,
                batchCount, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaEventRecord(t0, stream));
    for (int i = 0; i < 20; ++i) {
        if (layout == NN)
            cublasGemmStridedBatchedEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                &fa, dB, CUDA_R_16BF, N, (long long)K*N,
                     dA, CUDA_R_16BF, K, (long long)M*K,
                &fb, dC_ref, CUDA_R_16BF, N, (long long)M*N,
                batchCount, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        else if (layout == NT)
            cublasGemmStridedBatchedEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                &fa, dB, CUDA_R_16BF, K, (long long)N*K,
                     dA, CUDA_R_16BF, K, (long long)M*K,
                &fb, dC_ref, CUDA_R_16BF, N, (long long)M*N,
                batchCount, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        else
            cublasGemmStridedBatchedEx(cublas, CUBLAS_OP_N, CUBLAS_OP_T, N, M, K,
                &fa, dB, CUDA_R_16BF, N, (long long)K*N,
                     dA, CUDA_R_16BF, M, (long long)M*K,
                &fb, dC_ref, CUDA_R_16BF, N, (long long)M*N,
                batchCount, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
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

    double flops   = 2.0 * (double)M * N * K * batchCount;
    double blu_tfl = flops * 20.0 / (blu_ms * 1e-3) * 1e-12;
    double cub_tfl = flops * 20.0 / (cub_ms * 1e-3) * 1e-12;

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC_ref));
    CHECK_CUDA(cudaFree(dC_blu));
    CHECK_CUDA(cudaEventDestroy(t0));
    CHECK_CUDA(cudaEventDestroy(t1));

    return { blu_tfl, cub_tfl, rel_err, rel_err < 0.05 };
}

static void run_layout(
    const char* name,
    Layout layout,
    mycublasHandle_t myblas,
    cublasHandle_t cublas,
    cudaStream_t stream,
    int& passed, int& total)
{
    struct Shape { int M, N, K, batch; const char* label; };
    const Shape shapes[] = {
        {  128,  128,   64,  1, "tiny  128x128x64"},
        {  256,  256,  128,  1, "small 256x256x128"},
        {  512,  512,  128,  1, "mid   512x512x128"},
        { 1024, 1024,  128,  1, "large 1k x1k x128"},
        { 2048, 2048,  256,  1, "xl    2k x2k x256"},
        { 4096, 4096,  512,  1, "llm   4k x4k x512"},
        { 1024, 1024, 1024,  1, "sq    1k x1k x1k"},
        { 2048, 2048, 2048,  1, "sq    2k x2k x2k"},
    };
    const int N_SHAPES = (int)(sizeof(shapes) / sizeof(shapes[0]));

    printf("\n=== %s ===\n", name);
    printf("%-24s %6s %6s %6s  %-9s  %-9s  %-11s  %s\n",
           "Shape", "M", "N", "K", "BluBLAS", "cuBLAS", "MaxRelErr", "Status");
    printf("%s\n", std::string(85, '-').c_str());

    for (int i = 0; i < N_SHAPES; ++i) {
        const Shape& s = shapes[i];
        Result r = run_test(layout, s.M, s.N, s.K, s.batch,
                            1.0f, 0.0f, myblas, cublas, stream);
        printf("%-24s %6d %6d %6d  %7.2f T  %7.2f T  %9.2e  %s\n",
               s.label, s.M, s.N, s.K,
               r.blublas_tflops, r.cublas_tflops, r.rel_err,
               r.pass ? "PASS" : "FAIL");
        ++total;
        if (r.pass) ++passed;
    }
}

int main()
{
    int dev = 0;
    CHECK_CUDA(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
    printf("Device : %s  (SM %d.%d)\n", prop.name, prop.major, prop.minor);

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    mycublasHandle_t myblas;
    if (mycublasCreate(&myblas) != MYCUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "mycublasCreate failed\n");
        return EXIT_FAILURE;
    }
    mycublasSetStream(myblas, stream);

    cublasHandle_t cublas;
    CHECK_CUBLAS(cublasCreate(&cublas));
    CHECK_CUBLAS(cublasSetStream(cublas, stream));

    int passed = 0, total = 0;

    run_layout("NN  C=A[M,K]*B[K,N]",    NN, myblas, cublas, stream, passed, total);
    run_layout("NT  C=A[M,K]*B[N,K]^T",  NT, myblas, cublas, stream, passed, total);
    run_layout("TN  C=A[K,M]^T*B[K,N]",  TN, myblas, cublas, stream, passed, total);

    printf("\n%s\n", std::string(85, '=').c_str());
    printf("Overall: %d / %d PASSED\n", passed, total);

    mycublasDestroy(myblas);
    CHECK_CUBLAS(cublasDestroy(cublas));
    CHECK_CUDA(cudaStreamDestroy(stream));

    return (passed == total) ? EXIT_SUCCESS : EXIT_FAILURE;
}
