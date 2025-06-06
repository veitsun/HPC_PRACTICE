#pragma once

#include <cuda_runtime.h>

static cudaEvent_t start, stop;
static void startTimer() {
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
}

static float stopTimer() {
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds;
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return milliseconds;
}