#include <cstdio>
#include <iostream>
int main() {
  int deviceCount = 0;
  // 获取当前系统中可用的设备数量
  cudaError_t err = cudaGetDeviceCount(&deviceCount);

  if (err != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    return -1;
  }

  std::cout << "Number of CUDA devices: " << deviceCount << std::endl;

  // 遍历所有设备并打印设备的属性信息
  for (int device_id = 0; device_id < deviceCount; ++device_id) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    std::cout << "Device ID: " << device_id << std::endl;
    std::cout << "Device Name: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor
              << std::endl;
    std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024)
              << " MB" << std::endl;
    std::cout << "L2 Cache Size: " << prop.l2CacheSize / 1024 << " KB"
              << std::endl;
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock
              << std::endl;
    std::cout << "-------------------------------------" << std::endl;
  }

  return 0;
}