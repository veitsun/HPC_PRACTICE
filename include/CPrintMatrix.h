#ifndef CPRINTMATRIX_H
#define CPRINTMATRIX_H

class CPrintMatrix {
private:
public:
  void print(float *A, int N);
  void printMatrix(float *matrix, int nx, int ny);
  void printMatrixABC(float *A, float *B, float *C, int nx, int ny);
};

#endif