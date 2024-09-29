#include "include/CInitialData.h"
// #include "include/CPrintMatrix.h"
#include "include/Num.h"
#include "include/common.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cublas_v2.h>
#include <iostream>
using namespace std;
// ---------------------------------------------------------------------------cublas
void matMult_cublas(int M, int N, int K, float alpha, float *A, float *B,
                    float beta, float *C, cublasHandle_t cuHandle,
                    float *cublasRef, const int elemNum) {
  float *cublasdeviceA;
  float *cublasdeviceB;
  float *cublasdeviceC;
  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
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
  cudaEventRecord(start, 0);
  cublasSgemm(cuHandle, CUBLAS_OP_T, CUBLAS_OP_T, M, K, N, &a, cublasdeviceA, N,
              cublasdeviceB, K, &b, cublasdeviceC, M);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  printf("matMult_cublas Time elapsed %f sec\n", time);
  cublasGetVector(elemNum, sizeof(float), cublasdeviceC, 1, cublasRef, 1);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(cublasdeviceA);
  cudaFree(cublasdeviceB);
  cudaFree(cublasdeviceC);
}

int main(int argc, char **argv) {
  float *hostA;
  float *hostB;
  float *hostC;
  float *cublasRef;
  float alpha = 1.0;
  float beta = 1.0;
  // 给主机上的三个矩阵分配内存
  hostA = (float *)malloc(elemNum * sizeof(float));
  hostB = (float *)malloc(elemNum * sizeof(float));
  hostC = (float *)malloc(elemNum * sizeof(float));
  cublasRef = (float *)malloc(elemNum * sizeof(float));
  // 主机上的三个矩阵初始化数据
  CInitialData cinitialData;
  cinitialData.initialDataABC(hostA, hostB, hostC, nx, ny);
  memset(cublasRef, 0, elemNum * sizeof(float));
  // cout << "测试主机上的三个矩阵是否已经被初始化数据" << endl;
  // CPrintMatrix cprintmatrix;
  // cprintmatrix.printMatrixABC(hostA, hostB, hostC, nx, ny);
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
  cublasDestroy(cuHandle);
  free(hostA);
  free(hostB);
  free(hostC);
  return 0;
}