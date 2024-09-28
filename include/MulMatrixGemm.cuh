#include <cuda_runtime.h>

// 定义设备矩阵乘法函数
__global__ void MulMatrixGemmOnDevice(int M, int N, int K, float alpha,
                                      float *A, float *B, float beta, float *C);