#include "include/CGemmWithC.h"
#include "include/CInitialData.h"
#include "include/CPrintMatrix.h"
#include "include/common.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cublas_v2.h>
#include <iostream>
using namespace std;

__global__ void MulMatrixOnDevice(int M, int N, int K, float alpha, float *A,
                                  float *B, float beta, float *C) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  // printf("%f %f\n", A[row * N + k] , B[k * N + col])
  if (row < M && col < N) {
    float temp = 0.0;
    for (int k = 0; k < K; k++) {
      temp += A[row * N + k] * B[k * N + col];
      // printf("%f %f\n", A[row * N + k], B[k * N + col]);
    }
    C[row * N + col] = alpha * temp + beta * C[row * N + col];
  }
}

// ---------------------------------------------------------------------------cublas
void matMult_cublas(int M, int N, int K, float alpha, float *A, float *B,
                    float beta, float *C, cublasHandle_t cuHandle,
                    float *cublasRef, const int elemNum) {
  float *cublasdeviceA;
  float *cublasdeviceB;
  float *cublasdeviceC;
  double iStart, iElaps;
  // 在显存中为计算矩阵开辟空间
  CHECK(cudaMalloc((void **)&cublasdeviceA, elemNum * sizeof(float)));
  CHECK(cudaMalloc((void **)&cublasdeviceB, elemNum * sizeof(float)));
  CHECK(cudaMalloc((void **)&cublasdeviceC, elemNum * sizeof(float)));

  // 将主机上的数据拷贝到设备中

  cublasSetVector(elemNum, sizeof(float), A, 1, cublasdeviceA, 1);
  cublasSetVector(elemNum, sizeof(float), B, 1, cublasdeviceB, 1);
  cublasSetVector(elemNum, sizeof(float), C, 1, cublasdeviceC, 1);

  // 传递矩阵相乘中的参数，并执行内核函数，矩阵相乘
  float a = 1;
  float b = 0;
  iStart = seconds();
  cublasSgemm(cuHandle, CUBLAS_OP_T, CUBLAS_OP_T, M, K, N, &a, cublasdeviceA, N,
              cublasdeviceB, K, &b, cublasdeviceC, M);
  iElaps = seconds();
  printf("matMult_cublas Time elapsed %f sec\n", iElaps - iStart);

  cublasGetVector(elemNum, sizeof(float), cublasdeviceC, 1, cublasRef, 1);

  cudaFree(cublasdeviceA);
  cudaFree(cublasdeviceB);
  cudaFree(cublasdeviceC);
}

int main(int argc, char **argv) {
  float *hostA;
  float *hostB;
  float *hostC;
  float *hostRef;
  float *gpuRef;
  float *cublasRef;

  int nx = 5;
  int ny = 5;
  int elemNum = nx * ny;

  // 给主机上的三个矩阵分配内存
  hostA = (float *)malloc(elemNum * sizeof(float));
  hostB = (float *)malloc(elemNum * sizeof(float));
  hostC = (float *)malloc(elemNum * sizeof(float));
  hostRef = (float *)malloc(elemNum * sizeof(float));
  gpuRef = (float *)malloc(elemNum * sizeof(float));
  cublasRef = (float *)malloc(elemNum * sizeof(float));

  // 主机上的三个矩阵初始化数据
  CInitialData cinitialData;
  cinitialData.initialData(hostA, elemNum);
  cinitialData.initialData(hostB, elemNum);
  cinitialData.initialData(hostC, elemNum);
  memset(hostRef, 0, elemNum * sizeof(float));
  memset(gpuRef, 0, elemNum * sizeof(float));
  memset(cublasRef, 0, elemNum * sizeof(float));

  // 测试主机上的三个矩阵是否已经被初始化数据
  cout << "测试主机上的三个矩阵是否已经被初始化数据" << endl;
  CPrintMatrix cprintmatrix;
  cprintmatrix.printMatrix(hostA, elemNum, nx, ny);
  cprintmatrix.printMatrix(hostB, elemNum, nx, ny);
  cprintmatrix.printMatrix(hostC, elemNum, nx, ny);

  double iStart, iElaps;
  // -----------------------------------------------------------------------------------------
  // 在主机上执行矩阵乘法
  cout << "在主机上执行矩阵乘法" << endl;
  CGemmWithC girl;
  float alpha = 1.0;
  float beta = 1.0;
  iStart = seconds();
  girl.solveProblem(nx, nx, nx, alpha, hostA, hostB, beta, hostC, hostRef);
  iElaps = seconds();
  // girl.print(hostRef, elemNum); // 测试输出hostdef
  printf("MulMatrixOnHost Time elapsed %f sec\n", iElaps - iStart);

  cprintmatrix.printMatrix(hostA, elemNum, nx, ny);
  cprintmatrix.printMatrix(hostB, elemNum, nx, ny);
  cprintmatrix.printMatrix(hostC, elemNum, nx, ny);
  // printMatrix(hostRef, elemNum, nx, ny);
  // -----------------------------------------------------------------------------------------
  // 使用cuda kernel 来执行矩阵乘法
  // dim3 blockDim(elemNum / 8, elemNum / 8);
  // dim3 gridDim(8, 8);

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
  iStart = seconds();
  MulMatrixOnDevice<<<gridDim, blockDim>>>(nx, nx, nx, alpha, deviceA, deviceB,
                                           beta, deviceC);
  iElaps = seconds();
  // girl.print(gpuRef, elemNum);
  printf("MulMatrixOnDevice Time elapsed %f sec\n", iElaps - iStart);
  CHECK(cudaMemcpy(gpuRef, deviceC, elemNum * sizeof(float),
                   cudaMemcpyDeviceToHost));
  // girl.print(hostRef, elemNum);
  // girl.print(gpuRef, elemNum);
  // checkResult(hostRef, gpuRef, elemNum);
  CHECK(cudaDeviceSynchronize());
  // -----------------------------------------------------------------------------------------
  cout << "使用cublas 执行矩阵乘法" << endl;
  // 使用cublas 执行矩阵乘法

  // 创建并初始化cublas对象
  // 若是cublas对象在主函数中初始化，cublas方法在其他函数中调用，需要将cuHandle传入该函数，并在函数内创建status对象
  cublasHandle_t cuHandle;
  cublasStatus_t status = cublasCreate(&cuHandle);

  if (status != CUBLAS_STATUS_SUCCESS) {
    if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
      cout << "cublas 对象实例化出错" << endl;
    }
    getchar();
    return EXIT_FAILURE;
  }
  matMult_cublas(nx, nx, nx, alpha, hostA, hostB, beta, hostC, cuHandle,
                 cublasRef, elemNum);
  // checkResult(hostRef, cublasRef, elemNum);

  // 善后

  // printMatrix(hostA, elemNum, nx, ny);
  // printMatrix(hostB, elemNum, nx, ny);
  // printMatrix(hostC, elemNum, nx, ny);
  // printMatrix(hostRef, elemNum, nx, ny);
  // printMatrix(gpuRef, elemNum, nx, ny);

  CHECK(cudaFree(deviceA));
  CHECK(cudaFree(deviceB));
  CHECK(cudaFree(deviceC));
  cublasDestroy(cuHandle);
  free(hostA);
  free(hostB);
  free(hostC);
  free(hostRef);
  free(gpuRef);

  return 0;
}
