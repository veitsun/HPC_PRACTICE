#include "cuda_runtime.h"
#include <cublas_v2.h>

void AddKernel(int *a, int *b, int *c, int DX);