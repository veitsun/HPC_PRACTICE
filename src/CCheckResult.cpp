#include "CCheckResult.h"
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

void CCheckResult::compare(float *hostC, float *serialC, int M, int N) {
  float error = 0;
  bool tmp = true;
  for (int i = 0; i < M * N; i++) {
    error = fmax(error, fabs(hostC[i] - serialC[i]));
    if (error > 1e-5) {
      tmp = false;
      printf("error:hostC[%d] = %.3f, serialC[%d] = %.3f\n", i, hostC[i], i,
             serialC[i]);
      break;
    }
  }
  if (tmp) {
    printf("GPU output all right\n");
  }
}
