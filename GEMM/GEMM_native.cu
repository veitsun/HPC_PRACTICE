// #include <__clang_cuda_runtime_wrapper.h>
// #include <__clang_cuda_builtin_vars.h>
#include <cstdio>
#include <cuda_runtime.h>

__global__ void matMult(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    float sum = 0;
    for (int i = 0; i < K; i++) {
      sum += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = alpha * sum + beta * C[row * N + col];
  }
}

int main(int argc, char *argv[]) {
  float alpha = 1.0;
  float beta = 0.0;

  return 0;
}