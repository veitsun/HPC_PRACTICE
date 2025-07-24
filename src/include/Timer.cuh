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
  cudaEventSynchronize(stop); // 让宿主机线程等待 GPU 上的操作完成
  float milliseconds;
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return milliseconds;
}

// static cudaEvent_t start_stream, stop_stream;
static void startTimerStream(cudaEvent_t start_stream, cudaEvent_t stop_stream,
                             cudaStream_t &stream) {
  cudaEventCreate(&start_stream);
  cudaEventCreate(&stop_stream);
  cudaEventRecord(start_stream, stream);
}

static float stopTimerStream(cudaEvent_t start_stream, cudaEvent_t stop_stream,
                             cudaStream_t &stream) {
  cudaEventRecord(stop_stream, stream);
  cudaEventSynchronize(stop_stream); // 让宿主机线程等待 GPU 上的操作完成
  float milliseconds;
  cudaEventElapsedTime(&milliseconds, start_stream, stop_stream);
  cudaEventDestroy(start_stream);
  cudaEventDestroy(stop_stream);
  return milliseconds;
}