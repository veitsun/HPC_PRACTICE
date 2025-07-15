#include "CInitialData.h"
#include "CPrintMatrix.h"
#include "Num.h"
#include "Timer.cuh"
#include "common.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

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
  // 这对应的是当前线程块处理的整个部分 左上角起始位置的索引
  A += cRow * BLOCKSIZE * K;                    // row=cRow, col=0
  B += cCol * BLOCKSIZE;                        // row=0, col=cCol
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

  float tmp = 0.0;
  for (
      int bkIdx = 0; bkIdx < K;
      bkIdx +=
      BLOCKSIZE) { // 外层for循环是对线程块处理的部分进行分片迭代，即K维度上的循环迭代，在每次循环中，将当前线程块分片加载到共享内存中
    // Have each thread load one of the elements in A & B
    // Make the threadCol (=threadIdx.x) the consecutive index
    // to allow global memory access coalescing
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    // A, B的起始位置已经移动到分片的起始位置
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

    // block threads in this block until cache is fully populated
    __syncthreads();
    A += BLOCKSIZE;
    B += BLOCKSIZE * N; // 移动到下一个分片的起始位置？

    // execute the dotproduct on the currently cached block
    // 在当前的缓存分块上执行点积
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      // 是对当前线程块分片沿着K维度逐一进行点积（A行
      // B列），然后写入矩阵C对应位置
      // 每次内层for循环时，每个线程同时对应A分片dotIdx行中一个元素和B
      tmp += As[threadRow * BLOCKSIZE + dotIdx] *
             Bs[dotIdx * BLOCKSIZE + threadCol];
    }
    // need to sync again at the end, to avoid faster threads
    // fetching the next block into the cache before slower threads are done
    __syncthreads();
    // 需要在最后再次同步，以避免较快的线程在较慢的线程完成之前将下一个块提取到缓存中
  }
  C[threadRow * N + threadCol] =
      alpha * tmp + beta * C[threadRow * N + threadCol];
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

  // L1 cache becomes useless, since we access GMEM only via SMEM, so we carve
  // out all of L1 to SMEM. This doesn't currently make a difference, since
  // occupancy is limited by reg and thread count, but it's good to do anyway.
  cudaFuncSetAttribute(_sgemm_shared_mem_block<32>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  _sgemm_shared_mem_block<32>
      <<<gridDim, blockDim>>>(M, N, K, alpha, deviceA, deviceB, beta, deviceC);
  startTimer();
  for (int i = 0; i < repeat; i++) {
    _sgemm_shared_mem_block<32><<<gridDim, blockDim>>>(M, N, K, alpha, deviceA,
                                                       deviceB, beta, deviceC);
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