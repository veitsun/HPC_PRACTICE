#include "MulMatrixGemm.cuh"

__global__ void MulMatrixGemmOnDevice(int M, int N, int K, float alpha,
                                      float *A, float *B, float beta,
                                      float *C) {
  // 下标
  int row = threadIdx.y + gridDim.y * blockDim.y;
  int col = threadIdx.x + gridDim.x * blockDim.x;
  if (row < M && col < N) {
    float temp = 0.0;
    for (int k = 0; k < K; k++) {
      temp += A[k + row * K] * B[col + k * N];
    }
    C[row * N + col] = alpha * temp + beta * C[row * N + col];
  }
}

void MulMatrixGemmOnDeviceKernel(int M, int N, int K, float alpha, float *A,
                                 float *B, float beta, float *C, int matrixRow,
                                 int matrixCol) {
  dim3 blockDim(32, 32);
  dim3 gridDim((matrixCol + blockDim.x - 1) / blockDim.x,
               (matrixRow + blockDim.y - 1) / blockDim.y);
  MulMatrixGemmOnDevice<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  cudaDeviceSynchronize();
}