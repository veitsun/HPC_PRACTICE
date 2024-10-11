#include <cuda_runtime.h>
#include <iostream>

__global__ void add(int *a, int *b, int *c, int N) {
  // 创建共享内存数组，每个block的线程共享这段内存
  __shared__ int sharedA[32];
  __shared__ int sharedB[32];

  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < N) {
    // 将全局内存的数据加载到共享内存
    sharedA[threadIdx.x] = a[tid];
    sharedB[threadIdx.x] = b[tid];

    // __syncthreads()确保所有线程都完成了数据加载(对于一个block来说的)
    __syncthreads();

    // 执行元素加法
    c[tid] = sharedA[threadIdx.x] + sharedB[threadIdx.x];
  }
}

int main() {
  int N = 32;
  int size = N * sizeof(int);

  int h_a[N], h_b[N], h_c[N];
  int *d_a, *d_b, *d_c;

  // 初始化数组
  for (int i = 0; i < N; ++i) {
    h_a[i] = i;
    h_b[i] = i * 2;
  }

  // 分配设备内存
  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);

  // 复制数据到设备
  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

  // 每个block 32个线程（对应一个warp）
  add<<<1, 32>>>(d_a, d_b, d_c, N);

  // 将结果从设备复制回主机
  cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

  // 打印结果
  for (int i = 0; i < N; ++i) {
    std::cout << h_c[i] << " ";
  }
  std::cout << std::endl;

  // 释放设备内存
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
