#include "include/CGemmWithC.h"
#include "include/CInitialData.h"
#include "include/CPrintMatrix.h"
#include "include/Num.h"
#include "include/common.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cublas_v2.h>
#include <iostream>
using namespace std;

int main(int argc, char **argv) {
  float *hostA;
  float *hostB;
  float *hostC;
  float alpha = 1.0;
  float beta = 0.0;
  // 给主机上的三个矩阵分配内存
  hostA = (float *)malloc(elemNum * sizeof(float));
  hostB = (float *)malloc(elemNum * sizeof(float));
  hostC = (float *)malloc(elemNum * sizeof(float));
  // 主机上的三个矩阵初始化数据
  CInitialData cinitialData;
  cinitialData.initialDataABCByFile(hostA, hostB, hostC, n, n);

  cout << "测试主机上的三个矩阵是否已经被初始化数据" << endl;
  CPrintMatrix cprintmatrix;
  // cprintmatrix.printMatrixABC(hostA, hostB, hostC, nx, ny);

  // -----------------------------------------------------------------

  CGemmWithC girl;
  girl.solveProblem(M, N, K, alpha, hostA, hostB, beta, hostC);
  cprintmatrix.printMatrixCinFile(hostC, n, n);
  // std::cout << "C矩阵:" << std::endl;
  // cprintmatrix.printMatrix(hostC, nx, ny);

  free(hostA);
  free(hostB);
  free(hostC);
  return 0;
}