#include <cuda.h>
#include <iostream>

// PTX code as a string
const char *ptx_code = R"(
.version 8.5
.target sm_52
.address_size 64

.visible .entry _Z6mykernelPi(
    .param .u64 _Z6mykernelPi_param_0
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<3>;
    .ld.param.u64 %rd1, [_Z6mykernelPi_param_0];
    .cvta.to.global.u64 %rd2, %rd1;
    .mov.u32 %r1, %tid.x;
    .mul.lo.u32 %r2, %r1, 4;
    .add.s64 %rd3, %rd2, %r2;
    .st.global.u32 [%rd3], %r1;
    ret;
}
)";

int main() {
  CUdevice device;
  CUcontext context;
  CUmodule module;
  CUfunction kernel;
  CUresult res;

  // Initialize CUDA Driver API
  cuInit(0);
  cuDeviceGet(&device, 0);
  cuCtxCreate(&context, 0, device);

  // Load PTX module
  res = cuModuleLoadData(&module, ptx_code);
  if (res != CUDA_SUCCESS) {
    std::cerr << "Failed to load PTX module" << std::endl;
    return -1;
  }

  // Get kernel function from the module
  res = cuModuleGetFunction(&kernel, module, "_Z6mykernelPi");
  if (res != CUDA_SUCCESS) {
    std::cerr << "Failed to get kernel function" << std::endl;
    return -1;
  }

  // Allocate device memory
  int N = 64;
  int *d_data;
  cudaMalloc(&d_data, N * sizeof(int));

  // Launch kernel
  void *args[] = {&d_data};
  res = cuLaunchKernel(kernel, 1, 1, 1, N, 1, 1, 0, 0, args, 0);
  if (res != CUDA_SUCCESS) {
    std::cerr << "Failed to launch kernel" << std::endl;
    return -1;
  }

  // Cleanup
  cudaFree(d_data);
  cuModuleUnload(module);
  cuCtxDestroy(context);

  return 0;
}
