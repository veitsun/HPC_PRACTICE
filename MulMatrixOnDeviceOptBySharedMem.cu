#include "CInitialData.h"
#include "CPrintMatrix.h"
#include "Num.h"
#include "common.h"
// #include <__clang_cuda_builtin_vars.h>
// #include <__clang_cuda_builtin_vars.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cublas_v2.h>
using namespace std;

template <int BLOCK_DIM>
__global__ void MulMatrixOnDeviceOptBySharedMem(int M, int N, int K,
                                                float alpha, float *A, float *B,
                                                float beta, float *C) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float temp = 0.0f;
  __shared__ float sharedA[BLOCK_DIM][BLOCK_DIM];
  __shared__ float sharedB[BLOCK_DIM][BLOCK_DIM];
  int width = (K + BLOCK_DIM - 1) / BLOCK_DIM;

  for (int ph = 0; ph < width; ph++) {
    if (row < M && threadIdx.y + ph * BLOCK_DIM < K) {
      sharedA[threadIdx.y][threadIdx.x] =
          A[row * K + threadIdx.x + ph * BLOCK_DIM];
    } else {
      sharedA[threadIdx.y][threadIdx.x] = 0.0f;
    }
    if (col < N && threadIdx.x + ph * BLOCK_DIM < K) {
      sharedB[threadIdx.y][threadIdx.x] =
          B[(threadIdx.y + ph * BLOCK_DIM) * N + col];
    } else {
      sharedB[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();
    for (int s = 0; s < BLOCK_DIM; s++) {
      temp += sharedA[threadIdx.y][s] * sharedB[s][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = alpha * temp + beta * C[row * N + col];
  }
}

int main(int argc, char **argv) {
  float *hostA;
  float *hostB;
  float *hostC;
  float *gpuRef;
  float alpha = 1.0;
  float beta = 0.0;

  // 给主机上的三个矩阵分配内存
  hostA = (float *)malloc(elemNum * sizeof(float));
  hostB = (float *)malloc(elemNum * sizeof(float));
  hostC = (float *)malloc(elemNum * sizeof(float));
  gpuRef = (float *)malloc(elemNum * sizeof(float));
  // 主机上的三个矩阵初始化数据
  CInitialData cinitialData;
  cinitialData.initialDataABCByFile(hostA, hostB, hostC, n, n);
  memset(gpuRef, 0, elemNum * sizeof(float));

  CPrintMatrix cprintmatrix;
  // -------------------------------------------------------------------------------------GPU计时

  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // -----------------------------------------------------------------------------------------
  // 使用cuda kernel 来执行矩阵乘法
  dim3 blockDim(BLOCK_DIM_x, BLOCK_DIM_y);
  dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
               (n + blockDim.y - 1) / blockDim.y);
  float *deviceA;
  float *deviceB;
  float *deviceC;
  CHECK(cudaMalloc((float **)&deviceA, elemNum * sizeof(float)));
  CHECK(cudaMalloc((float **)&deviceB, elemNum * sizeof(float)));
  CHECK(cudaMalloc((float **)&deviceC, elemNum * sizeof(float)));
  CHECK(cudaMemcpy(deviceA, hostA, elemNum * sizeof(float),
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(deviceB, hostB, elemNum * sizeof(float),
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(deviceC, hostC, elemNum * sizeof(float),
                   cudaMemcpyHostToDevice));

  MulMatrixOnDeviceOptBySharedMem<32>
      <<<gridDim, blockDim>>>(n, n, n, alpha, deviceA, deviceB, beta, deviceC);
  cudaEventRecord(start, 0);
  // =----------------------------------------------------------------------------------------
  int repeat = 20;
  for (int i = 0; i < repeat; i++) {
    MulMatrixOnDeviceOptBySharedMem<32><<<gridDim, blockDim>>>(
        n, n, n, alpha, deviceA, deviceB, beta, deviceC);
  }

  // =----------------------------------------------------------------------------------------
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&time, start, stop);
  printf("MulMatrixOnDeviceOptBySharedMem Time elapsed %f ms\n", time / repeat);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  CHECK(cudaMemcpy(gpuRef, deviceC, elemNum * sizeof(float),
                   cudaMemcpyDeviceToHost));
  CHECK(cudaDeviceSynchronize());
  cprintmatrix.printMatrixCinFile(gpuRef, n, n);
  // -----------------------------------------------------------------------------------------
  CHECK(cudaFree(deviceA));
  CHECK(cudaFree(deviceB));
  CHECK(cudaFree(deviceC));
  free(hostA);
  free(hostB);
  free(hostC);
  free(gpuRef);

  return 0;
}