#ifndef UTIL_H
#define UTIL_H
// #include <__clang_cuda_builtin_vars.h>
#include <cuda.h>
__global__ void initKernel(float *data, float value, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    data[idx] = value;
  }
}

void init_two_arrary_mem_by_cpu(float *out_d, float *in_d, const int n,
                                float valueA, float valueB) {
  for (int i = 0; i < n; ++i) {
    out_d[i] = valueA;
    in_d[i] = valueB;
  }
}

__global__ void init_two_array_mem_by_gpu(float *out_d, float *in_d,
                                          const int n, float valueA,
                                          float valueB) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out_d[idx] = valueA;
    in_d[idx] = valueB;
  }
}

#endif