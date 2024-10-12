#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>

__global__ void add(int *a, int *b, int *c) {
  int i = threadIdx.x;
  c[i] = a[i] + b[i];
}

int main() {
  cudaDeviceProp prop;
  int device_id = 0;
  cudaGetDeviceProperties(&prop, device_id);
  size_t size =
      std::min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
  cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                     size); /* set-aside 3/4 of L2 cache for persisting accesses
                               or the max allowed*/
  const int N = 10;
  int *hostA;
  int *hostB;
  int *hostC;
  hostA = (int *)malloc(sizeof(int) * N);
  hostB = (int *)malloc(sizeof(int) * N);
  hostC = (int *)malloc(sizeof(int) * N);
  for (int i = 0; i < N; i++) {
    hostA[i] = i;
    hostB[i] = i;
    hostC[i] = 0;
  }
  int *deviceA, *deviceB, *deviceC;
  cudaMalloc((void **)&deviceA, N * sizeof(int));
  cudaMalloc((void **)&deviceB, sizeof(int) * N);
  cudaMalloc((void **)&deviceC, sizeof(int) * N);
  cudaMemcpy(deviceA, hostA, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceC, hostC, N * sizeof(int), cudaMemcpyHostToDevice);
  // 执行内核函数
  dim3 blockDim(N);
  dim3 gridDim(1);
  add<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC);

  cudaMemcpy(hostC, deviceC, N * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < N; i++) {
    std::cout << hostC[i] << " ";
  }
  std::cout << std::endl;
  // cudaMalloc()
  return 0;
}