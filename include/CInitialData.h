#ifndef CInitialData_H
#define CInitialData_H
#include "CMulMatrixGemm.h"

class CInitialData {
public:
  void initialData(float *ip, int size);
  void initialDataxy(float *ip, int nx, int ny);
  void initialDataABC(float *A, float *B, float *C, int nx, int ny);
  void initialMatrixGemmData(CMulMatrixGemm girl);
  void initialDataABCByFile(float *A, float *B, float *C, int nx, int ny);
  void initialDataABCByFileNames(float *A, float *B, float *C, int nx, int ny,
                                 const char *FILENAME);
};

#endif