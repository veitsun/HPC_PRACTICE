#include "include/CInitialData.h"
#include "include/CPrintMatrix.h"
#include "include/Num.h"
#include "include/common.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cublas_v2.h>
// #include <iostream>
using namespace std;

// #define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

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

// 用于计算每个线程的多个结果的一维块分片
template <const int BM, const int BN, const int BK, const int TM>
__global__ void _sgemm1DBlocktiling(int M, int N, int K, float alpha,
                                    const float *A, const float *B, float beta,
                                    float *C) {
  // If we flip x and y here we get ~30% less performance for large matrices.
  // The current, 30% faster configuration ensures that blocks with sequential
  // blockIDs access columns of B sequentially, while sharing the same row of A.
  // The slower configuration would share columns of A, but access into B would
  // be non-sequential. So the faster configuration has better spatial locality
  // and hence a greater L2 hit rate.
  const uint cRow = blockIdx.y; // cRou 和 cCol 块索引
  const uint cCol = blockIdx.x;

  // each warp will calculate 32*TM elements, with 32 being the columnar dim.
  // 每个warp将计算 32 * TM 元素，其中 32 是柱状维度
  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  // allocate space for the current blocktile in SMEM
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // todo: adjust this to each thread to load multiple entries and
  // better exploit the cache sizes
  assert(BM * BK == blockDim.x);
  assert(BN * BK == blockDim.x);
  // 用于确保某些条件在cuda核函数中成立，通常是在进行块和线程配置时验证参数的合理性
  const uint innerColA = threadIdx.x % BK; // warp-level GMEM coalescing
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColB = threadIdx.x % BN; // warp-level GMEM coalescing
  const uint innerRowB = threadIdx.x / BN;

  // allocate thread-local cache for results in registerfile
  float threadResults[TM] = {0.0};

  // outer loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    // advance blocktile
    A += BK;
    B += BK * N;

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // we make the dotproduct loop the outside loop, which facilitates
      // reuse of the Bs entry, which we can cache in a tmp var.
      float tmpB = Bs[dotIdx * BN + threadCol];
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * N + threadCol] =
        alpha * threadResults[resIdx] +
        beta * C[(threadRow * TM + resIdx) * N + threadCol];
  }
}

// 通过二维块分片增强计算强度
// kernel 5
// 的基本思想是每个线程计算C的8*8个元素的网格，内核的第一阶段是让所有的线程一起工作来填充SMEM缓存。我们将让每个线程加载多个元素
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    _sgemm2DBlocktiling(int M, int N, int K, float alpha, const float *A,
                        const float *B, float beta, float *C) {
  // __launch_bounds__((BM*BN) / (TM*TN), 1) 这个修饰符指定了内核的启动约束。
  // BM 和 BN 是块的维度（行和列） ， TM 和 TN 是线程的维度（行和列）
  // 这个表达式计算每个块的最大线程数，并限制每个块的线程数，帮助优化调度
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint totalResultsBlocktile = BM * BN; // 一个分块元素总数量
  // A thread is responsible for calculating TM*TN elements in the blocktile
  // 一个线程负责计算blocktile中的TM * TN个元素
  const uint numThreadsBlocktile =
      totalResultsBlocktile /
      (TM * TN); // 需要多少个线程来处理掉（一个分块中）共享内存里的全部数据

  // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
  assert(numThreadsBlocktile ==
         blockDim.x); // 每个block需要用到的线程数量是不是等于块中的线程总数

  // BN/TN are the number of threads to span a column. BN/TN
  // 是跨越一行需要的线程数
  const int threadCol = threadIdx.x % (BN / TN); // 列的线程索引(抽象的)
  const int threadRow = threadIdx.x / (BN / TN); // 行的线程索引(抽象的)

  // allocate space for the current blocktile in smem
  // 在共享内存中给当前的块片分配空间
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column移动到索引位置
  // 这里每个block的cRow和cCol是一样的,也就是说每block的ABC起始位置都是一样的
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // calculating the indices that this thread will load into SMEM
  // 计算被加载到共享内存的，这个分块对于整体所在的位置
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  // calculates the number of rows of As that are being loaded in a single step
  // by a single block
  const uint strideA =
      numThreadsBlocktile / BK; // 对应分块A中，一共可以使用的线程数量

  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;
  // for both As and Bs we want each load to span the full column-width, for
  // better GMEM coalescing (as opposed to spanning full row-width and iterating
  // across columns)
  // 对于A和B，我们希望每次加载都能跨越整个列宽，以便更好地进行全局内存合并（而不是跨越整个行宽并跨列迭代）
  const uint strideB =
      numThreadsBlocktile / BN; // 对应分块B中，一共可以使用的线程数量

  // allocate thread-local cache for results in registerfile  为
  // 寄存器文件中的结果 分配 线程本地缓存
  float threadResults[TM * TN] = {0.0};
  // register caches for As and Bs 给AS和BS的寄存器缓存
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles 最外层循环遍历方块
  for (uint bkIdx = 0; bkIdx < K;
       bkIdx += BK) { // block索引依次遍历，宽度是BK，分块地宽度是BK
    // populate the SMEM caches 填充共享内存缓存
    for (uint loadOffset = 0; loadOffset < BM;
         loadOffset += strideA) { // 将全局内存中的数据加载到共享内存中
      As[(innerRowA + loadOffset) * BK + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }
    for (uint loadOffset = 0; loadOffset < BK;
         loadOffset += strideB) { // 将全局内存中的数据加载到共享内存中
      Bs[(innerRowB + loadOffset) * BN + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }
    __syncthreads();
    // 现在共享内存缓存已经填充完毕，我们让每个线程将其相关的共享内存条目相乘，并将结果累计到本地寄存器中。

    // advance blocktile
    A += BK;     // move BK columns to right 将A首地址移动到右边
    B += BK * N; // move BK rows down 将B的首地址移动到下面

    // calculate per-thread results 计算每个线程的结果
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // block into registers  将As和Bs相关的条目加载到寄存器中
      // 两个for i循环将重复使用的线程分片元素加载至寄存器。
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
        // 将As的共享内存中的数据加载到寄存器中
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
        // 将Bs的共享内存中的数据加载到寄存器中
      }
      // 在寄存器缓存中计算外积，把结果放进结果数组中
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
      // 在内层循环中，我们可以通过将dotIdx作为外层循环，并将两个内层循环所需的值显式加载到寄存器中，来减少对共享内存的访问次数
    }
    __syncthreads();
  }

  // write out the results 将结果写回
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
          alpha * threadResults[resIdxM * TN + resIdxN] +
          beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
    }
  }
}

// 向量化 SMEM 和 GMEM 访问    、、 向量化共享内存和全局内存访问
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void _sgemmVectorize(int M, int N, int K, float alpha, float *A,
                                float *B, float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // allocate space for the current blocktile in smem
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // calculating the indices that this thread will load into SMEM
  // we'll load 128bit / 32bit = 4 elements per thread at each step
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);

  // allocate thread-local cache for results in registerfile
  float threadResults[TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    // transpose A while loading it
    float4 tmp =
        reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

    reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
        reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // block into registers
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[dotIdx * BM + threadRow * TM + i];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
    for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
      // load C vector into registers
      float4 tmp = reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
      // perform GEMM update in reg
      tmp.x = alpha * threadResults[resIdxM * TN + resIdxN] + beta * tmp.x;
      tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
      tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
      tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;
      // write back
      reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
          tmp;
    }
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
  // dim3 blockDim(BLOCK_DIM_x, BLOCK_DIM_y);
  // dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
  //              (n + blockDim.y - 1) / blockDim.y);
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
  int repeat = 20;
  // // 朴素的矩阵乘法
  // MulMatrixOnDevice<<<gridDim, blockDim>>>(n, n, n, alpha, deviceA, deviceB,
  //                                          beta, deviceC);

  // // 全局内存合并
  // _sgemm_global_mem_coalesce<32>
  //     <<<gridDim, blockDim>>>(n, n, n, alpha, deviceA, deviceB, beta,
  //     deviceC);

  // // 共享内存缓存分块
  // _sgemm_shared_mem_block<32>
  //     <<<gridDim, blockDim>>>(n, n, n, alpha, deviceA, deviceB, beta,
  //     deviceC);

  // // 用于计算每个线程的多个结果的一维块分片
  // const uint BM = 64;
  // const uint BN = 64;
  // const uint BK = 8;
  // const uint TM = 8;
  // dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  // dim3 blockDim((BM * BN) / TM);
  // _sgemm1DBlocktiling<BM, BN, BK, TM>
  //     <<<gridDim, blockDim>>>(M, N, K, alpha, deviceA, deviceB, beta,
  //     deviceC);

  // 通过二维块分片增加计算强度
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  // if (M >= 128 and N >= 128) {
  const uint BM = 128;
  const uint BN = 128;
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim((BM * BN) / (TM * TN));
  _sgemm2DBlocktiling<BM, BN, BK, TM, TN>
      <<<gridDim, blockDim>>>(M, N, K, alpha, deviceA, deviceB, beta, deviceC);
  // } else {
  //   // this is a hacky solution to the underlying problem
  //   // of not having proper bounds checking in the kernel
  //   const uint BM = 64;
  //   const uint BN = 64;
  //   dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  //   dim3 blockDim((BM * BN) / (TM * TN));
  //   _sgemm2DBlocktiling<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(
  //       M, N, K, alpha, deviceA, deviceB, beta, deviceC);
  // }

  // --------------------------------------------------------------------------------------------kernel开始计时
  cudaEventRecord(start, 0);
  // --------------------------------------------------------------------------------------------kernel

  for (int i = 0; i < repeat; i++) {
    // // 朴素的矩阵乘法
    // MulMatrixOnDevice<<<gridDim, blockDim>>>(n, n, n, alpha, deviceA,
    // deviceB,
    //                                          beta, deviceC);

    // // 全局内存合并
    // _sgemm_global_mem_coalesce<32><<<gridDim, blockDim>>>(
    //     n, n, n, alpha, deviceA, deviceB, beta, deviceC);

    // // 共享内存缓存分块
    // _sgemm_shared_mem_block<32><<<gridDim, blockDim>>>(n, n, n, alpha,
    // deviceA,
    //                                                    deviceB, beta,
    //                                                    deviceC);

    // // 用于计算每个线程的多个结果的一维块分片
    // _sgemm1DBlocktiling<64, 64, 8, 8><<<(n * n + 511) / 512, 512>>>(
    //     M, N, K, alpha, deviceA, deviceB, beta, deviceC);

    // 通过二维块分片增加计算强度
    _sgemm2DBlocktiling<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(
        M, N, K, alpha, deviceA, deviceB, beta, deviceC);

  } // MulMatrixOnDevice
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