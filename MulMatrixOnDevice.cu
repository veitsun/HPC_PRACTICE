#include "include/CInitialData.h"
// #include "include/CPrintMatrix.h"
#include "include/Num.h"
#include "include/common.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cublas_v2.h>
// #include <iostream>
using namespace std;

__global__ void MulMatrixOnDevice(int M, int N, int K, float alpha, float *A,
                                  float *B, float beta, float *C) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    float temp = 0.0;
    for (int k = 0; k < K; k++) {
      temp += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = alpha * temp + beta * C[row * N + col];
  }
}

int main(int argc, char **argv) {
  float *hostA;
  float *hostB;
  float *hostC;
  float *hostRef;
  float *gpuRef;
  float alpha = 1.0;
  float beta = 1.0;

  int elemNum = nx * ny;

  // 给主机上的三个矩阵分配内存
  hostA = (float *)malloc(elemNum * sizeof(float));
  hostB = (float *)malloc(elemNum * sizeof(float));
  hostC = (float *)malloc(elemNum * sizeof(float));
  hostRef = (float *)malloc(elemNum * sizeof(float));
  gpuRef = (float *)malloc(elemNum * sizeof(float));

  // 主机上的三个矩阵初始化数据
  CInitialData cinitialData;
  cinitialData.initialData(hostA, elemNum);
  cinitialData.initialData(hostB, elemNum);
  cinitialData.initialData(hostC, elemNum);
  memset(hostRef, 0, elemNum * sizeof(float));
  memset(gpuRef, 0, elemNum * sizeof(float));

  // 测试主机上的三个矩阵是否已经被初始化数据
  // cout << "测试主机上的三个矩阵是否已经被初始化数据" << endl;
  // CPrintMatrix cprintmatrix;
  // cprintmatrix.printMatrix(hostA, elemNum, nx, ny);
  // cprintmatrix.printMatrix(hostB, elemNum, nx, ny);
  // cprintmatrix.printMatrix(hostC, elemNum, nx, ny);

  // -------------------------------------------------------------------------------------GPU计时

  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // -----------------------------------------------------------------------------------------
  // 使用cuda kernel 来执行矩阵乘法
  dim3 blockDim(16, 16);
  dim3 gridDim((ny + blockDim.x - 1) / blockDim.x,
               (nx + blockDim.y - 1) / blockDim.y);
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
  cudaEventRecord(start, 0);
  MulMatrixOnDevice<<<gridDim, blockDim>>>(nx, nx, nx, alpha, deviceA, deviceB,
                                           beta, deviceC);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&time, start, stop);
  printf("MulMatrixOnDevice Time elapsed %f sec\n", time);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  CHECK(cudaMemcpy(gpuRef, deviceC, elemNum * sizeof(float),
                   cudaMemcpyDeviceToHost));
  CHECK(cudaDeviceSynchronize());
  // -----------------------------------------------------------------------------------------
  // 善后
  // printMatrix(hostA, elemNum, nx, ny);
  // printMatrix(hostB, elemNum, nx, ny);
  // printMatrix(hostC, elemNum, nx, ny);
  // printMatrix(gpuRef, elemNum, nx, ny);

  CHECK(cudaFree(deviceA));
  CHECK(cudaFree(deviceB));
  CHECK(cudaFree(deviceC));
  free(hostA);
  free(hostB);
  free(hostC);
  free(hostRef);
  free(gpuRef);

  return 0;
}