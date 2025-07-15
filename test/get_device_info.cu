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
    if (!prop.deviceOverlap) {
      printf("Your Device will not support speed up from multi-streams\n");
      return 0;
    }

    std::cout << "Device ID: " << device_id << std::endl;
    std::cout << "Device Name: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor
              << std::endl;
    std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024)
              << " MB" << std::endl;
    std::cout << "L2 Cache Size: " << prop.l2CacheSize / 1024 << " KB"
              << std::endl; // GPU上可用的L2缓存量
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock
              << std::endl;

    std::cout
        << "可留存内存访问预留的最大 L2 缓存量:"
        << prop.persistingL2CacheMaxSize
        << std::
               endl; // 这里是0，可能是在某些的GPU架构中，L2缓存的使用方式可能有所不同，例如动态分配而非静态预留。因此为0，不一定意味着无法利用L2缓存来加速数据访问。
    std::cout << "访问策略窗口的最大大小:" << prop.accessPolicyMaxWindowSize
              << std::endl;
    std::cout << "-------------------------------------" << std::endl;
  }

  return 0;
}