#include <iostream>
#include <vector>
#include <iomanip>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            std::cerr << "CUDA error at " << __FILE__ << ":"        \
                      << __LINE__ << " code=" << err << " \""       \
                      << cudaGetErrorString(err) << "\"" << std::endl; \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

#define CHECK_CUBLAS(call)                                          \
    do {                                                            \
        cublasStatus_t status = call;                               \
        if (status != CUBLAS_STATUS_SUCCESS) {                      \
            std::cerr << "cuBLAS error at " << __FILE__ << ":"      \
                      << __LINE__ << " code=" << status << std::endl; \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

struct TestSize {
    int m, n, k;
};

void benchmark_cublas_bf16(cublasHandle_t cublas, int m, int n, int k, int warmup, int iters) {
    size_t size_A = (size_t)m * k * sizeof(__nv_bfloat16);
    size_t size_B = (size_t)k * n * sizeof(__nv_bfloat16);
    size_t size_C = (size_t)m * n * sizeof(__nv_bfloat16);

    __nv_bfloat16 *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, size_A));
    CHECK_CUDA(cudaMalloc(&dB, size_B));
    CHECK_CUDA(cudaMalloc(&dC, size_C));

    // Initialize with zeros (just for benchmarking latency)
    CHECK_CUDA(cudaMemset(dA, 0, size_A));
    CHECK_CUDA(cudaMemset(dB, 0, size_B));
    CHECK_CUDA(cudaMemset(dC, 0, size_C));

    float alpha = 1.0f;
    float beta = 0.0f;

    // cuBLAS interprets matrices as column-major.
    // To compute C = A * B (row-major), we compute C^T = B^T * A^T (col-major).
    cublasOperation_t transA = CUBLAS_OP_N; // B^T in col-major is just B
    cublasOperation_t transB = CUBLAS_OP_N; // A^T in col-major is just A

    int lda = n;
    int ldb = k;
    int ldc = n;

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        CHECK_CUBLAS(cublasGemmEx(
            cublas, transA, transB,
            n, m, k, // M and N are swapped for C^T = B^T * A^T
            &alpha,
            dB, CUDA_R_16BF, lda,
            dA, CUDA_R_16BF, ldb,
            &beta,
            dC, CUDA_R_16BF, ldc,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        CHECK_CUBLAS(cublasGemmEx(
            cublas, transA, transB,
            n, m, k,
            &alpha,
            dB, CUDA_R_16BF, lda,
            dA, CUDA_R_16BF, ldb,
            &beta,
            dC, CUDA_R_16BF, ldc,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= iters;

    double tflops = (2.0 * m * n * k) / (ms * 1e9);

    std::cout << "  M=" << std::setw(5) << m 
              << " N=" << std::setw(5) << n 
              << " K=" << std::setw(5) << k
              << " | Latency: " << std::fixed << std::setprecision(3) << std::setw(8) << ms << " ms"
              << " | TFLOPS: " << std::fixed << std::setprecision(2) << std::setw(8) << tflops 
              << std::endl;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
}

int main() {
    int dev = 0;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
    std::cout << "Device: " << prop.name << " (SM " << prop.major << "." << prop.minor << ")\n\n";

    cublasHandle_t cublas;
    CHECK_CUBLAS(cublasCreate(&cublas));
    CHECK_CUBLAS(cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH));

    std::vector<TestSize> sizes = {
        {2048, 2048, 2048},
        {4096, 4096, 4096},
        {8192, 8192, 8192},
        {16384, 16384, 16384},
        {16384, 4096, 4096},
        {4096, 16384, 4096},
        {4096, 4096, 16384}
    };

    std::cout << "cuBLAS BF16 GEMM Benchmark:\n";
    std::cout << "=================================================================\n";
    for (const auto& s : sizes) {
        benchmark_cublas_bf16(cublas, s.m, s.n, s.k, 5, 20);
    }
    std::cout << "=================================================================\n";

    CHECK_CUBLAS(cublasDestroy(cublas));
    return 0;
}
