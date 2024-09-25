#include "../include/CGemmWithC.h"
#include <cstdio>
#include <iostream>

using namespace std;
void CGemmWithC::solveProblem(int M, int N, int K, float alpha, float *A,
                              float *B, float beta, float *C, float *hostRef) {
  printf("M,N,K %d:%d:%d\n", M, N, K);
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0;
      for (int k = 0; k < K; k++) {
        //
        sum += A[i * N + k] * B[k * N + j];
      }
      // C[i * N + j] = sum * alpha + beta * C[i * N + j];
      // hostRef[i * N + j] = C[i *N + j];
      hostRef[i * N + j] = sum * alpha + beta * C[i * N + j];
    }
  }
  std::cout << std::endl;
}
