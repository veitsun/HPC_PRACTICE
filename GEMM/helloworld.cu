#include "CInitialData.h"
#include "CPrintMatrix.h"
#include "Num.h"
#include "common.h"
#include "myBase.cuh"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cublas_v2.h>

__global__ void helloFromGPU() { printf("Hello World from GPU!\n"); }

int main() {
  helloFromGPU<<<1, 10>>>();
  cudaDeviceReset();
  return 0;
}