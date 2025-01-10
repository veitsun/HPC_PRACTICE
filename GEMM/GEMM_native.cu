// #include <__clang_cuda_runtime_wrapper.h>
// #include <__clang_cuda_builtin_vars.h>
#include "CInitialData.h"
#include "CPrintMatrix.h"
#include "Num.h"
#include "common.h"
#include "myBase.cuh"
// #include <__clang_cuda_builtin_vars.h>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// 朴素的实现方式
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

  float *hostA;
  float *hostB;
  float *hostC;

  hostA = (float *)malloc(elemNum * sizeof(float));
  hostB = (float *)malloc(elemNum * sizeof(float));
  hostC = (float *)malloc(elemNum * sizeof(float));

  CInitialData cinitialdata;
  // cinitialdata.initialDataABCByFile(hostA, hostB, hostC, n, n);
  cinitialdata.initialDataABCByFileNames(hostA, hostB, hostC, n, n,
                                         "./data/random_numbers.txt");

  float *deviceA;
  float *deviceB;
  float *deviceC;

  CHECK(cudaMalloc((void **)&deviceA, elemNum * sizeof(float)));
  CHECK(cudaMalloc((void **)&deviceB, elemNum * sizeof(float)));
  CHECK(cudaMalloc((void **)&deviceC, elemNum * sizeof(float)));

  CHECK(cudaMemcpy(deviceA, hostA, elemNum, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(deviceB, hostB, elemNum, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(deviceC, hostC, elemNum, cudaMemcpyHostToDevice));

  dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
  dim3 gridDim((n + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
               (n + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

  int repeat = 20;
  // 朴素的矩阵乘法
  matMult<<<gridDim, blockDim>>>(M, n, K, alpha, deviceA, deviceB, beta,
                                 deviceC);

  startTimer();
  for (int i = 0; i < repeat; i++) {
    matMult<<<gridDim, blockDim>>>(M, n, K, alpha, deviceA, deviceB, beta,
                                   deviceC);
  }
  float time = stopTimer();
  CHECK(cudaDeviceSynchronize());

  printf("朴素矩阵乘法 Time elapsed %f ms\n", time / repeat);

  CHECK(cudaMemcpy(hostC, deviceC, elemNum, cudaMemcpyDeviceToHost));
  CPrintMatrix cprintmatrix;
  // cprintmatrix.printMatrixCinFile(hostC, n, n);
  cprintmatrix.printMatrixCinFileByNames(hostC, n, n, "./data/result.txt");
  CHECK(cudaFree(deviceA));
  CHECK(cudaFree(deviceB));
  CHECK(cudaFree(deviceC));
  free(hostA);
  free(hostB);
  free(hostC);
  return 0;
}