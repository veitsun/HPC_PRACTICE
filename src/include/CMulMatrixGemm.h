#ifndef CMULMATRIXGEMM_H
#define CMULMATRIXGEMM_H

#include "MulMatrixGemm.cuh"

class CMulMatrixGemm {
private:
  int sumSize;

public:
  float *A;
  float *B;
  float *C;
  int M;
  int N;
  int K;
  float alpha;
  float beta;

  void setSumSize(int num);
  int getSumSize();
  void initData();
  void mulMatrixGemm();
  void show();
};

#endif