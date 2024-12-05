#include "AddNumkernel.cuh"

// 内核函数
__global__ void Add(int *a, int *b, int *c, int DX) {
  int f = blockIdx.x * blockDim.x + threadIdx.x;

  if (f >= DX)
    return;

  c[f] = a[f] + b[f];
}

// 调用内核函数
void AddKernel(int *a, int *b, int *c, int DX) {
  dim3 dimBlock = (128);
  dim3 dimGrid = ((DX + 128 - 1) / 128);
  Add<<<dimGrid, dimBlock>>>(a, b, c, DX);
  cudaDeviceSynchronize();
}