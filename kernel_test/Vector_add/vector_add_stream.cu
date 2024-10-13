#include <cuda_runtime.h>
#include <iostream>

// CUDA 内核函数，进行简单的向量加法
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    C[i] = A[i] + B[i];
  }
}

int main() {
  // 定义向量大小
  int N = 1 << 20; // 1M 个元素
  size_t size = N * sizeof(float);

  // 在主机 (Host) 上分配内存
  float *h_A = (float *)malloc(size);
  float *h_B = (float *)malloc(size);
  float *h_C = (float *)malloc(size);

  // 初始化向量 A 和 B
  for (int i = 0; i < N; ++i) {
    h_A[i] = static_cast<float>(i);
    h_B[i] = static_cast<float>(i * 2);
  }

  // 在设备 (Device) 上分配内存
  float *d_A, *d_B, *d_C;
  cudaMalloc((void **)&d_A, size);
  cudaMalloc((void **)&d_B, size);
  cudaMalloc((void **)&d_C, size);

  // 创建两个 CUDA Stream
  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  // 设置线程和块的数量
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  // 使用 Stream1：异步将 A 数组从主机传输到设备
  cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream1);

  // 使用 Stream2：异步将 B 数组从主机传输到设备
  cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream2);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float time;
  cudaEventRecord(start, 0);
  // 使用 Stream1：在设备上启动向量加法内核
  vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_A, d_B, d_C, N);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  printf("Vector_add_stream Time elapsed %f ms\n", time);

  // 使用 Stream1：异步将结果 C 从设备传输回主机
  cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream1);

  // 等待所有 Stream 执行完成
  cudaStreamSynchronize(stream1);
  cudaStreamSynchronize(stream2);

  // 检查结果
  bool success = true;
  for (int i = 0; i < N; ++i) {
    if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5) {
      success = false;
      break;
    }
  }
  if (success) {
    std::cout << "Vector addition completed successfully!" << std::endl;
  } else {
    std::cout << "Vector addition failed!" << std::endl;
  }

  // 释放资源
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}
