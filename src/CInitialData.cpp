#include "../include/CInitialData.h"
#include "../include/common.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>
void CInitialData::initialData(float *ip, int size) {
  // generate different seed for random number
  time_t t;
  srand((unsigned)time(&t));

  for (int i = 0; i < size; i++) {
    ip[i] = (float)(rand() & 0xFF) / 10.0f;
  }

  return;
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