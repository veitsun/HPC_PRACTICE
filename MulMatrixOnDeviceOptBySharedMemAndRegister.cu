#include "CInitialData.h"
#include "CPrintMatrix.h"
#include "Num.h"
#include "common.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cublas_v2.h>
using namespace std;

const int TM = 4;
const int TN = 4;
const int BM = TM * BLOCK_DIM_X;
const int BN = TN * BLOCK_DIM_Y;
const int BK = 8;

template <int BM, int BN, int BK, int TM, int TN>
__global__ void matrixKernel1st(float *dA, float *dB, float *dC, int M, int K,
                                int N) {
  __shared__ float SA[BM * BK];
  __shared__ float SB[BK * BN];
  int indA = TM * (threadIdx.x + blockIdx.x * blockDim.x);
  int indB = TN * (threadIdx.y + blockIdx.y * blockDim.y);
  int width = (K + BK - 1) / BK;
  float tmp[TM * TN] = {0.0f};

  for (int ph = 0; ph < width; ph++) {

    for (int index_q = 0; index_q < TM; index_q++) {
      for (int index_k = 0; index_k < BK; index_k++) {
        if (indA + index_q < M && index_k + ph * BK < K) {
          SA[(threadIdx.x * TM + index_q) * BK + index_k] =
              dA[(indA + index_q) * K + index_k + ph * BK];
        } else {
          SA[(threadIdx.x * TM + index_q) * BK + index_k] = 0.0f;
        }
      }
    }
    __syncthreads();
    for (int index_v = 0; index_v < TN; index_v++) {
      for (int index_k = 0; index_k < BK; index_k++) {

        if (indB + index_v < N && index_k + ph * BK < K) {

          SB[index_k * BN + threadIdx.y * TN + index_v] =
              dB[(index_k + ph * BK) * N + indB + index_v];
        } else {
          SB[index_k * BN + threadIdx.y * TN + index_v] = 0.0f;
        }
      }
    }

    __syncthreads();
    for (int index_q = 0; index_q < TM; index_q++) {
      for (int index_v = 0; index_v < TN; index_v++) {
        for (int index_k = 0; index_k < BK; index_k++) {
          tmp[index_q * TN + index_v] +=
              SA[(threadIdx.x * TM + index_q) * BK + index_k] *
              SB[index_k * BN + threadIdx.y * TN + index_v];
        }
      }
    }
    __syncthreads();
  }
  for (int index_q = 0; index_q < TM; index_q++) {
    for (int index_v = 0; index_v < TN; index_v++) {
      if (indA + index_q < M && indB + index_v < N) {
        dC[(indA + index_q) * N + indB + index_v] = tmp[index_q * TN + index_v];
      }
    }
  }
}
template <int BM, int BN, int BK, int TM, int TN>
__global__ void matrixKernel2nd(float *dA, float *dB, float *dC, int M, int K,
                                int N) {
  __shared__ float SA[BM * BK];
  __shared__ float SB[BK * BN];
  int indA = TM * (blockIdx.x * blockDim.x);
  int indB = TN * (blockIdx.y * blockDim.y);
  int width = (K + BK - 1) / BK;
  float tmp[TM * TN] = {0.0f};
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int smem_a_m = tid % 128;
  int smem_a_k = tid / 128;
  int smem_b_k = tid % 8;
  int smem_b_n = tid / 8;
  for (int ph = 0; ph < width; ph++) {

    if (indA + smem_a_m < M && smem_a_k + ph * BK < K) {
      SA[smem_a_m * BK + smem_a_k] =
          dA[(indA + smem_a_m) * K + smem_a_k + ph * BK];
    } else {
      SA[smem_a_m * BK + smem_a_k] = 0.0f;
    }
    if (indB + smem_b_n < N && smem_b_k + ph * BK < K) {

      SB[smem_b_k * BN + smem_b_n] =
          dB[(smem_b_k + ph * BK) * N + indB + smem_b_n];
    } else {
      SB[smem_b_k * BN + smem_b_n] = 0.0f;
    }

    __syncthreads();
    for (int index_q = 0; index_q < TM; index_q++) {
      for (int index_v = 0; index_v < TN; index_v++) {
        int reg_c_m = threadIdx.x * TM + index_q;
        int reg_c_n = threadIdx.y * TN + index_v;
        for (int index_k = 0; index_k < BK; index_k++) {
          tmp[index_q * TN + index_v] +=
              SA[reg_c_m * BK + index_k] * SB[index_k * BN + reg_c_n];
        }
      }
    }
    __syncthreads();
  }
  for (int index_q = 0; index_q < TM; index_q++) {
    for (int index_v = 0; index_v < TN; index_v++) {
      int reg_c_m = threadIdx.x * TM + index_q;
      int reg_c_n = threadIdx.y * TN + index_v;
      if (indA + index_q < M && indB + index_v < N) {
        dC[(indA + reg_c_m) * N + indB + reg_c_n] = tmp[index_q * TN + index_v];
      }
    }
  }
}
template <int BM, int BN, int BK, int TM, int TN>
__global__ void matrixOrigin(float *dA, float *dB, float *dC, int M, int K,
                             int N) {

  int indA = TM * (threadIdx.x + blockIdx.x * blockDim.x);
  int indB = TN * (threadIdx.y + blockIdx.y * blockDim.y);
  float tmp[TM][TN] = {0.0f};
  for (int index_q = 0; index_q < TM; index_q++) {
    for (int index_v = 0; index_v < TN; index_v++) {
      if (indA + index_q < M && indB + index_v < N) {
        for (int s = 0; s < K; s++) {
          tmp[index_q][index_v] +=
              dA[(indA + index_q) * K + s] * dB[s * N + indB + index_v];
        }
      }
    }
  }
  for (int index_q = 0; index_q < TM; index_q++) {
    for (int index_v = 0; index_v < TN; index_v++) {
      if (indA + index_q < M && indB + index_v < N) {
        dC[(indA + index_q) * N + indB + index_v] = tmp[index_q][index_v];
      }
    }
  }
}

void hostMatrix(float *hostA, float *hostB, float *hostC, int M, int K, int N) {
  float *dA, *dB, *dC;
  cudaMalloc((void **)&dA, M * K * sizeof(float));
  cudaMalloc((void **)&dB, N * K * sizeof(float));
  cudaMalloc((void **)&dC, M * N * sizeof(float));

  cudaMemcpy(dA, hostA, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hostB, N * K * sizeof(float), cudaMemcpyHostToDevice);

  int num_blocks_x = (M + BM - 1) / BM;
  int num_blocks_y = (N + BN - 1) / BN;
  dim3 block_dim(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
  dim3 grid_dim(num_blocks_x, num_blocks_y, 1);
  int repeat = 20;
  matrixKernel1st<BM, BN, BK, TM, TN>
      <<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
  // matrixKernel2nd<BM, BN, BK, TM, TN>
  //     <<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
  // matrixOrigin<BM, BN, BK, TM, TN>
  //     <<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
  cudaEvent_t start, stop;
  float ker_time = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  for (int i = 0; i < repeat; i++) {
    // matrixKernel1st<BM, BN, BK, TM, TN>
    //     <<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
    // matrixKernel2nd<BM, BN, BK, TM, TN>
    //     <<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
    matrixOrigin<BM, BN, BK, TM, TN>
        <<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
    // Possibly: exit(-1) if program cannot continue....
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time

  cudaMemcpy(hostC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  CPrintMatrix cprintmatrix;
  cprintmatrix.printMatrixCinFileClear(hostC, n, n);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  printf("M-K-N: %d-%d-%d\n", M, K, N);
  printf("kernel time: %.4f second, %.4f ms\n", ker_time / (repeat * 1000.),
         ker_time / repeat);
  printf("grid dim: %d, %d, %d\n", grid_dim.x, grid_dim.y, grid_dim.z);
  printf("block dim: %d, %d, %d\n", block_dim.x, block_dim.y, block_dim.z);
}

int main(int argc, char **argv) {
  float *hostA;
  float *hostB;
  float *hostC;
  // 给主机上的三个矩阵分配内存
  hostA = (float *)malloc(M * K * sizeof(float));
  hostB = (float *)malloc(N * K * sizeof(float));
  hostC = (float *)malloc(M * N * sizeof(float));
  // 主机上的三个矩阵初始化数据
  CInitialData cinitialData;
  cinitialData.initialDataABC(hostA, hostB, hostC, n, n);
  // -----------------------------------------------------------------------------------------
  hostMatrix(hostA, hostB, hostC, M, K, N);
  free(hostA);
  free(hostB);
  free(hostC);
  return 0;
}