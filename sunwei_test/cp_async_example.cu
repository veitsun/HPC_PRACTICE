/**
 * @file cp_async_example.cu
 * @author maverick (you@domain.com)
 * @brief 使用 cp.async 指令来实现数据从全局内存异步拷贝到共享内存的案例
 * @version 0.1
 * @date 2025-02-20
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "common.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>
__global__ void kernel(float *d_out, const float *d_in) {
  __shared__ float s_data[4];

  // 异步拷贝数据到共享内存
  // cp.async.ca.shared.global指令将数据从全局内存异步拷贝到共享内存
  // cuda 的 内联汇编，
  // 指示编译器将包含的汇编代码嵌入到生成的机器代码中，并防止对其进行优化。
  // asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
  //              : // 无输出
  //              : "l"(s_data), "l"(d_in));

  // // 等待拷贝完成
  // asm volatile("cp.async.wait_group 0;\n");

  int idx = threadIdx.x;
  if (idx < 4) {
    s_data[idx] = d_in[idx];
  }
  __syncthreads();

  // 进行计算

  if (idx < 4) {
    s_data[idx] = s_data[idx] * 2.0f; // 示例计算：将共享内存中的数据乘以2
  }

  // 将结果写回全局内存
  if (idx < 4) {
    d_out[idx] = s_data[idx];
  }
}

int main(int argc, char *argv[]) {
  float *d_in, *d_out;
  cudaMalloc(&d_in, 4 * sizeof(float));
  cudaMalloc(&d_out, 4 * sizeof(float));
  float *host_in = (float *)malloc(4 * sizeof(float));
  float *host_out = (float *)malloc(4 * sizeof(float));
  for (int i = 0; i < 4; i++) {
    host_in[i] = i;
  }

  CHECK(cudaMemcpy(d_in, host_in, 4 * sizeof(float), cudaMemcpyHostToDevice));

  kernel<<<1, 4>>>(d_out, d_in);

  CHECK(cudaMemcpy(host_out, d_out, 4 * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < 4; i++) {
    printf("%f ", host_out[i]);
  }
  std::cout << std::endl;

  cudaFree(d_in);
  cudaFree(d_out);
  free(host_in);
  free(host_out);

  return 0;
}