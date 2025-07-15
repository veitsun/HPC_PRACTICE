// #include <__clang_cuda_runtime_wrapper.h>
// #include <__clang_cuda_builtin_vars.h>
#include "CInitialData.h"
#include "CPrintMatrix.h"
#include "Num.h"
#include "Timer.cuh"
#include "common.h"
// #include <__clang_cuda_builtin_vars.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

// 朴素的实现方式
// __global__ void matMult(int M, int N, int K, float alpha, float *A, float *B,
//                         float beta, float *C) {
//   int row = blockIdx.y * blockDim.y + threadIdx.y;
//   int col = blockIdx.x * blockDim.x + threadIdx.x;
//   if (row < M && col < N) {
//     float sum = 0.0;
//     for (int i = 0; i < K; i++) {
//       sum += A[row * K + i] * B[i * N + col];
//     }
//     C[row * N + col] = alpha * sum + beta * C[row * N + col];
//   }
// }
__global__ void matMult(int M, int N, int K, float alpha, const float *A,
                        const float *B, float beta, float *C) {
  const uint row = blockIdx.x * blockDim.x + threadIdx.x;
  const uint col = blockIdx.y * blockDim.y + threadIdx.y;
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

  // CHECK(cudaMemcpy(deviceA, hostA, elemNum * sizeof(float),
  //                  cudaMemcpyHostToDevice));
  // CHECK(cudaMemcpy(deviceB, hostB, elemNum * sizeof(float),
  //                  cudaMemcpyHostToDevice));
  // CHECK(cudaMemcpy(deviceC, hostC, elemNum * sizeof(float),
  //                  cudaMemcpyHostToDevice));
  cudaStream_t stream1, stream2, stream3;
  CHECK(cudaStreamCreate(&stream1));
  CHECK(cudaStreamCreate(&stream2));
  CHECK(cudaStreamCreate(&stream3));

  CHECK(cudaMemcpyAsync(deviceA, hostA, elemNum * sizeof(float),
                        cudaMemcpyHostToDevice, stream1));
  CHECK(cudaMemcpyAsync(deviceB, hostB, elemNum * sizeof(float),
                        cudaMemcpyHostToDevice, stream2));
  CHECK(cudaMemcpyAsync(deviceC, hostC, elemNum * sizeof(float),
                        cudaMemcpyHostToDevice, stream3));

  // 同步每个流，确保流中的所有操作都执行完成
  CHECK(cudaStreamSynchronize(stream1));
  CHECK(cudaStreamSynchronize(stream2));
  CHECK(cudaStreamSynchronize(stream3));

  dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
  dim3 gridDim((n + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
               (n + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

  int repeat = 20;
  // 朴素的矩阵乘法
  matMult<<<gridDim, blockDim>>>(M, N, K, alpha, deviceA, deviceB, beta,
                                 deviceC);

  startTimer();
  for (int i = 0; i < repeat; i++) {
    matMult<<<gridDim, blockDim>>>(M, N, K, alpha, deviceA, deviceB, beta,
                                   deviceC);
  }
  float time = stopTimer();
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(gpuRef, deviceC, elemNum * sizeof(float),
                   cudaMemcpyDeviceToHost));
  CPrintMatrix cprintmatrix;
  printf("朴素矩阵乘法 Time elapsed %f ms\n", time / repeat);

  cprintmatrix.printMatrixCinFileByNames(
      gpuRef, n, n, "./data/output_data/result_native.txt");
  CHECK(cudaFree(deviceA));
  CHECK(cudaFree(deviceB));
  CHECK(cudaFree(deviceC));
  free(hostA);
  free(hostB);
  free(hostC);
  free(gpuRef);
  return 0;
}