// #include <__clang_cuda_builtin_vars.h>
// #include <__clang_cuda_builtin_vars.h>
#include "utils.cuh"
#include <cstdint>
#include <cuda.h>
#define N 500000

__global__ void shortKernel(float *out_d, float *in_d) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    out_d[idx] = in_d[idx] * 2.0f;
  }
}

#define NSTEP 1000
#define NKERNEL 20

const int thread_per_block = 256;
const int block_per_grid = (N + thread_per_block - 1) / thread_per_block;

int main() {
  float *out_d = nullptr;
  float *in_d = nullptr;
  cudaMalloc(&out_d, N * sizeof(float));
  cudaMalloc(&in_d, N * sizeof(float));
  const int valueA = 0.1;
  const int valueB = 0.5;
  // 两种初始化形式
  init_mem_by_cpu(out_d, in_d, N, valueA, valueB); // 方式一，用 CPU 进行初始化

  init_mem_by_gpu<<<block_per_grid, thread_per_block>>>(
      out_d, in_d, N, valueA, valueB); // 方式二，用 GPU 进行初始化

  for (int istep = 0; istep < NSTEP; ++istep) {
    for (int ikernel = 0; ikernel < NKERNEL; ++ikernel) {
      shortKernel<<<block_per_grid, thread_per_block>>>(out_d, in_d);
    }
  }

  cudaFree(out_d);
  cudaFree(in_d);
  return 0;
}