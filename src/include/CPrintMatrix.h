#ifndef CPRINTMATRIX_H
#define CPRINTMATRIX_H

class CPrintMatrix {
private:
public:
  void print(float *C, int N);
  void printMatrix(float *matrix, int nx, int ny);
  void printMatrixABC(float *A, float *B, float *C, int nx, int ny);
  void printMatrixCinFile(float *C, int nx, int ny);
  void printMatrixCinFileClear(float *C, int nx, int ny);
  void printMatrixCinFileByNames(float *C, int nx, int ny,
                                 const char *FILENAME);
};

#endif