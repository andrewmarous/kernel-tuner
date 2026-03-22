#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C,
        int M, int N, int K)
{
    int k = blockDim.x * blockIdx.x + threadIdx.y; // column index
    int m = blockDim.y * blockIdx.y + threadIdx.y; // row index
    int idx = m * K + k;

    if ((k < K) && (m < M))
    {
        float sum = 0;
        for (int n = 0; n < N; ++n)
        {
            sum += A[m*N + n] * B[n*K + k];
        }
        C[idx] += sum;
    }
}
