#include "include/CInitialData.h"
#include "include/CPrintMatrix.h"
#include "include/Num.h"
#include "include/common.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cublas_v2.h>
// #include <iostream>
using namespace std;

// 朴素实现
__global__ void MulMatrixOnDevice(int M, int N, int K, float alpha,
                                  const float *A, const float *B, float beta,
                                  float *C) {
  const uint row = blockIdx.y * blockDim.y + threadIdx.y;
  const uint col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    float temp = 0.0;
    for (int k = 0; k < K; k++) {
      temp += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = alpha * temp + beta * C[row * N + col];
  }
}

// 全局内存合并
template <const uint BLOCKSIZE>
__global__ void _sgemm_global_mem_coalesce(int M, int N, int K, float alpha,
                                           const float *A, const float *B,
                                           float beta, float *C) {
  const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  // if statement is necessary to make things work under tile quantization
  if (cRow < M && cCol < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[cRow * K + i] * B[i * N + cCol];
    }
    C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
  }
}

// 共享内存缓存分块
template <const int BLOCKSIZE>
__global__ void _sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                        const float *A, const float *B,
                                        float beta, float *C) {
  // the output block that we want to compute in this threadblock
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  // allocate buffer for current block in fast shared mem
  // shared mem is shared between all threads in a block
  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  // the inner row & col that we're accessing in this thread
  const uint threadCol = threadIdx.x % BLOCKSIZE;
  const uint threadRow = threadIdx.x / BLOCKSIZE;

  // advance pointers to the starting positions
  A += cRow * BLOCKSIZE * K;                    // row=cRow, col=0
  B += cCol * BLOCKSIZE;                        // row=0, col=cCol
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

  float tmp = 0.0;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    // Have each thread load one of the elements in A & B
    // Make the threadCol (=threadIdx.x) the consecutive index
    // to allow global memory access coalescing
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

    // block threads in this block until cache is fully populated
    __syncthreads();
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    // execute the dotproduct on the currently cached block
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      tmp += As[threadRow * BLOCKSIZE + dotIdx] *
             Bs[dotIdx * BLOCKSIZE + threadCol];
    }
    // need to sync again at the end, to avoid faster threads
    // fetching the next block into the cache before slower threads are done
    __syncthreads();
  }
  C[threadRow * N + threadCol] =
      alpha * tmp + beta * C[threadRow * N + threadCol];
}

int main(int argc, char **argv) {
  float *hostA;
  float *hostB;
  float *hostC;
  float *gpuRef;

  float alpha = 1.0;
  float beta = 0.0;

  // 给主机上的三个矩阵分配内存
  hostA = (float *)malloc(M * K * sizeof(float));
  hostB = (float *)malloc(K * N * sizeof(float));
  hostC = (float *)malloc(M * N * sizeof(float));
  gpuRef = (float *)malloc(M * N * sizeof(float));

  // 主机上的三个矩阵初始化数据
  CInitialData cinitialData;
  cinitialData.initialDataABCByFile(hostA, hostB, hostC, n, n);
  memset(gpuRef, 0, M * N * sizeof(float));
  // -------------------------------------------------------------------------------------GPU计时

  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // -----------------------------------------------------------------------------------------
  // 使用 cuda kernel 来执行矩阵乘法
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
  cudaEventRecord(start, 0);
  int repeat = 20;
  // // 朴素的矩阵乘法
  // MulMatrixOnDevice<<<gridDim, blockDim>>>(n, n, n, alpha, deviceA, deviceB,
  //                                          beta, deviceC);

  // 全局内存合并
  _sgemm_global_mem_coalesce<32>
      <<<gridDim, blockDim>>>(n, n, n, alpha, deviceA, deviceB, beta, deviceC);
  // --------------------------------------------------------------------------------------------kernel

  for (int i = 0; i < repeat; i++) {
    // // 朴素的矩阵乘法
    // MulMatrixOnDevice<<<gridDim, blockDim>>>(n, n, n, alpha, deviceA,
    // deviceB,
    //                                          beta, deviceC);

    // 全局内存合并
    _sgemm_global_mem_coalesce<32><<<gridDim, blockDim>>>(
        n, n, n, alpha, deviceA, deviceB, beta, deviceC);
  }
  // --------------------------------------------------------------------------------------------kernel
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
    // Possibly: exit(-1) if program cannot continue....
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&time, start, stop);
  printf("MulMatrixOnDevice Time elapsed %f ms\n", time / repeat);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  CHECK(cudaMemcpy(gpuRef, deviceC, elemNum * sizeof(float),
                   cudaMemcpyDeviceToHost));
  CHECK(cudaDeviceSynchronize());
  CPrintMatrix cprintmatrix;
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