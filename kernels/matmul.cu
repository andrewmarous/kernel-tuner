#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

// AGENT-TUNABLE PARAMS
#define MATMUL_BLOCK_SIZE_X 16
#define MATMUL_BLOCK_SIZE_Y 16
// END AGENT-TUNABLE PARAMS

// KERNEL
__global__ void matmul_kernel(const float* A, const float* B, float* C,
        int M, int N, int K)
{
    int k = blockDim.x * MATMUL_BLOCK_SIZE_X + threadIdx.x; // col index
    int m = blockDim.y * MATMUL_BLOCK_SIZE_Y + threadIdx.y; // row index

    if ((k < K) && (m < M))
    {
        float sum = 0.0f;
        for (int n = 0; n < N; ++n)
        {
            sum += A[m*N + n] * B[n*K + k];
        }
        C[m*K + k] = sum;
    }
}

#ifndef CUDA_CHECK
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#endif
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void run_test(int M, int N, int K) {
    size_t size_A = M * N * sizeof(float);
    size_t size_B = N * K * sizeof(float);
    size_t size_C = M * K * sizeof(float);

    float *h_A, *h_B, *h_C;
    CUDA_CHECK(cudaMallocManaged(&h_A, size_A));
    CUDA_CHECK(cudaMallocManaged(&h_B, size_B));
    CUDA_CHECK(cudaMallocManaged(&h_C, size_C));

    // Fill with random data
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0, 1.0);
    for(int i=0; i < M*N; ++i) h_A[i] = dist(gen);
    for(int i=0; i < N*K; ++i) h_B[i] = dist(gen);
    std::fill_n(h_C, M*K, 0.0f);

    // Grid config
    dim3 threads(MATMUL_BLOCK_SIZE_X, MATMUL_BLOCK_SIZE_Y);
    dim3 blocks((K + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_kernel<<<blocks, threads>>>(h_A, h_B, h_C, M, N, K);
    cudaEventRecord(stop);

    CUDA_CHECK(cudaDeviceSynchronize());

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Size " << M << "x" << N << "x" << K << " | Time: " << milliseconds << " ms" << std::endl;

    // Cleanup
    cudaFree(h_A); cudaFree(h_B); cudaFree(h_C);
}

int main() {
    std::cout << "Starting Performance Tests..." << std::endl;
    run_test(128, 128, 128); // Small
    run_test(512, 512, 512); // Medium
    run_test(1024, 1024, 1024); // Large
    return 0;
}
