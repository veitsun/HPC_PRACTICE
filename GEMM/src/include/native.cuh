#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

//  二维 block + 二维 grid （最原始的矩阵乘法实现） M K * K N = M N
__global__ void native_matmul(int M, int N, int K, const float *A, const float *B, float *C) {
	const uint row = threadIdx.x + blockDim.x * blockIdx.x;
	const uint col = threadIdx.y + blockDim.y * blockIdx.y;
	if(row < M && col < N) {
		float temp = 0.0;
		for(int i = 0; i < K; i ++) {
			// temp += A[row][i] * B[i][col];
			temp += A[row * K + i] * B[i * N + col];
		}
		C[row * N + col] += temp;
	}
}

// 二维 block + 二维 grid （原始的矩阵乘法 + 合并访存）
__global__ void native_matmul_version2(int M, int N, int K, const float *A, const float *B, float *C) {
	const uint row = threadIdx.y + blockDim.y * blockIdx.y;
	const uint col = threadIdx.x + blockDim.x * blockIdx.x;
	if(row < M && col < N) {
		float temp = 0.0;
		for(int i = 0; i < K; i ++) {
			temp += A[row * K + i] * B[i * N + col];
		}
		C[row * N + col] += temp;
	}
}