#include "../include/CPrintMatrix.h"
#include <cstdio>

void CPrintMatrix::print(float *A, int N) {
  for (int i = 0; i < N; i++) {
    printf("%f ", *(A + i));
  }
  printf("\n");
}

void CPrintMatrix::printMatrix(float *matrix, int size, int nx, int ny) {
  printf("Matrix: \n");
  float *A = matrix;
  for (int i = 0; i < size; i++) {

    printf("%f ", *(A + i));
    if ((i + 1) % ny == 0) {
      printf("\n");
    }
  }
  printf("\n\n");
}
