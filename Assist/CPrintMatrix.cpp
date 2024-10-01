#include "../include/CPrintMatrix.h"
#include <cstdio>
#include <fstream>
#include <iostream>

void CPrintMatrix::print(float *C, int N) {
  for (int i = 0; i < N; i++) {
    printf("%f ", *(C + i));
  }
  printf("\n");
}

void CPrintMatrix::printMatrix(float *matrix, int nx, int ny) {
  int size = nx * ny;
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

void CPrintMatrix::printMatrixABC(float *A, float *B, float *C, int nx,
                                  int ny) {
  std::cout << "hostA ";
  printMatrix(A, nx, ny);
  std::cout << "hostB ";
  printMatrix(B, nx, ny);
  std::cout << "hostC ";
  printMatrix(C, nx, ny);
}

void CPrintMatrix::printMatrixCinFile(float *C, int nx, int ny) {
  std::ofstream outfile;
  outfile.open("fileoutput.txt", std::ios::out);
  int size = nx * ny;
  // printf("Matrix:\n");
  outfile << "Matrix C:" << std::endl;
  for (int i = 0; i < size; i++) {
    outfile << *(C + i) << " ";
    if ((i + 1) % ny == 0) {
      outfile << std::endl;
    }
  }
  outfile << std::endl;
  outfile.close();
}
