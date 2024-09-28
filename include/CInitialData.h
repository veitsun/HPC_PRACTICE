#ifndef CInitialData_H
#define CInitialData_H
#include "CMulMatrixGemm.h"

class CInitialData {
public:
  void initialData(float *ip, int size);
  void initialMatrixGemmData(CMulMatrixGemm girl);
};

#endif