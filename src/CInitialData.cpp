#include "../include/CInitialData.h"
#include "../include/common.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
void CInitialData::initialData(float *ip, int size) {
  // generate different seed for random number
  time_t t;
  srand((unsigned)time(&t));

  for (int i = 0; i < size; i++) {
    ip[i] = (float)(rand() & 0xFF) / 10.0f;
  }

  return;
}

void CInitialData::initialDataxy(float *ip, int nx, int ny) {
  int size = nx * ny;
  // generate different seed for random number
  time_t t;
  srand((unsigned)time(&t));

  for (int i = 0; i < size; i++) {
    ip[i] = (float)(rand() & 0xFF) / 10.0f;
  }

  return;
}

void CInitialData::initialDataABC(float *A, float *B, float *C, int nx,
                                  int ny) {
  int elemNum = nx * ny;
  initialData(A, elemNum);
  initialData(B, elemNum);
  initialData(C, elemNum);
}

void CInitialData::initialDataABCByFile(float *A, float *B, float *C, int nx,
                                        int ny) {
  const char *FILENAME = "random_numbers.txt"; // 文件名
  const int MAX_NUMBERS = nx * ny;
  std::ifstream inputFile(FILENAME);
  if (!inputFile.is_open()) {
    std::cerr << "无法打开文件 " << FILENAME << std::endl;
    return;
  }
  int count = 0;
  float number;
  while (inputFile >> number && count < MAX_NUMBERS) {
    *(A + count) = number;
    *(B + count) = number;
    *(C + count) = number;
    count++;
  }
  inputFile.close();

  // initialData(A, elemNum);
  // initialData(B, elemNum);
  // initialData(C, elemNum);
  // std::ifstream fin("file")
}

void CInitialData::initialMatrixGemmData(CMulMatrixGemm girl) {
  // 这里的代码有问题，还没有进行更改
  int elemNum = girl.getSumSize();
  float *deviceA;
  float *deviceB;
  float *deviceC;
  cudaMalloc((float **)&deviceA, elemNum * sizeof(float));
  cudaMalloc((float **)&deviceB, elemNum * sizeof(float));
  cudaMalloc((float **)&deviceC, elemNum * sizeof(float));
  cudaMemcpy(deviceA, girl.A, elemNum * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, girl.B, elemNum * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceC, girl.C, elemNum * sizeof(float), cudaMemcpyHostToDevice);
}