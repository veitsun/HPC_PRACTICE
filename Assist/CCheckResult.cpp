#include "../include/CCheckResult.h"
#include <cmath>
#include <cstdio>

void CCheckResult::checkResult(float *hostRef, float *gpuRef, const int N) {
  double epsilon = 1.0E-8;
  bool match = 1;

  for (int i = 0; i < N; i++) {
    if (fabs(hostRef[i] - gpuRef[i]) > epsilon) {
      match = 0;
      printf("Arrays do not match!\n");
      printf("%d\n", i);
      printf("host %5.7f gpu %5.7f at current %d\n", hostRef[i], gpuRef[i], i);
      break;
    }
  }

  if (match)
    printf("Arrays match.\n\n");

  return;
}