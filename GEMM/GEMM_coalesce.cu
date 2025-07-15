#include "CInitialData.h"
#include "CPrintMatrix.h"
#include "Num.h"
#include "Timer.cuh"
#include "common.h"
// #include <__clang_cuda_builtin_vars.h>
// #include <__clang_cuda_builtin_vars.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

template <const uint BLOCK_SIZE>
__global__ void gemm_global_mem_coalesce(int M, int N, int K, float alpha,
                                         const float *A, const float *B,
                                         float beta, float *C) {
  const uint row = blockIdx.x * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE);
  const uint col = blockIdx.y * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);
  if (row < M && col < N) {
    float temp = 0.0;
    for (int k = 0; k < K; k++) {
      temp += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = alpha * temp + beta * C[row * N + col];
  }
}

int main(int argc, char *argv[]) {
  float alpha = 1.0;
  float beta = 0.0;

  float *hostA;
  float *hostB;
  float *hostC;
  float *gpuRef;

  hostA = (float *)malloc(M * K * sizeof(float));
  hostB = (float *)malloc(K * N * sizeof(float));
  hostC = (float *)malloc(M * N * sizeof(float));
  gpuRef = (float *)malloc(M * N * sizeof(float));
  memset(gpuRef, 0, M * N * sizeof(float));

  CInitialData cinitialdata;
  cinitialdata.initialDataABCByFileNames(hostA, hostB, hostC, n, n,
                                         INPUTFILENAME.c_str());

  // CPrintMatrix cc;
  // cc.printMatrix(hostA, n, n);
  // cc.printMatrix(hostB, n, n);
  // cc.printMatrix(hostC, n, n);
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

  dim3 blockDim(BLOCK_DIM_X * BLOCK_DIM_Y);
  dim3 gridDim((n + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
               (n + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

  int repeat = 20;
  // 朴素的矩阵乘法
  gemm_global_mem_coalesce<32>
      <<<gridDim, blockDim>>>(M, N, K, alpha, deviceA, deviceB, beta, deviceC);

  startTimer();
  for (int i = 0; i < repeat; i++) {
    gemm_global_mem_coalesce<32><<<gridDim, blockDim>>>(M, N, K, alpha, deviceA,
                                                        deviceB, beta, deviceC);
  }
  float time = stopTimer();
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(gpuRef, deviceC, elemNum * sizeof(float),
                   cudaMemcpyDeviceToHost));
  CPrintMatrix cprintmatrix;
  printf("朴素矩阵乘法 Time elapsed %f ms\n", time / repeat);

  // cprintmatrix.printMatrixCinFile(hostC, n, n);
  cprintmatrix.printMatrixCinFileByNames(
      gpuRef, n, n, "./data/output_data/result_coalesce.txt");
  CHECK(cudaFree(deviceA));
  CHECK(cudaFree(deviceB));
  CHECK(cudaFree(deviceC));
  free(hostA);
  free(hostB);
  free(hostC);
  free(gpuRef);
  return 0;
}