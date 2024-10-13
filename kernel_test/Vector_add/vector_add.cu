// #include <algorithm>
// #include <__clang_cuda_builtin_vars.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>

__global__ void add(float *a, float *b, float *c, int N) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  c[i] = a[i] + b[i];
}

int main() {
  const int N = 1 << 20;
  float *hostA;
  float *hostB;
  float *hostC;
  hostA = (float *)malloc(sizeof(float) * N);
  hostB = (float *)malloc(sizeof(float) * N);
  hostC = (float *)malloc(sizeof(float) * N);
  for (int i = 0; i < N; ++i) {
    hostA[i] = static_cast<float>(i);
    hostB[i] = static_cast<float>(i * 2);
  }
  float *deviceA, *deviceB, *deviceC;
  cudaMalloc((void **)&deviceA, N * sizeof(int));
  cudaMalloc((void **)&deviceB, sizeof(int) * N);
  cudaMalloc((void **)&deviceC, sizeof(int) * N);
  cudaMemcpy(deviceA, hostA, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceC, hostC, N * sizeof(int), cudaMemcpyHostToDevice);
  // 执行内核函数
  // 设置线程和块的数量
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float time;
  cudaEventRecord(start, 0);

  add<<<blocksPerGrid, threadsPerBlock>>>(deviceA, deviceB, deviceC, N);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  printf("Vector_add Time elapsed %f ms\n", time);

  // cudaMemcpy(hostC, deviceC, N * sizeof(int), cudaMemcpyDeviceToHost);
  // for (int i = 0; i < N; i++) {
  //   std::cout << hostC[i] << " ";
  // }
  std::cout << std::endl;
  free(hostA);
  free(hostB);
  free(hostC);
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  return 0;
}