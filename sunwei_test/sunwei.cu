#include <cstdio>

__global__ void add(int *a, int *b, int *c) {
  int i = threadIdx.x;
  c[i] = a[i] + b[i];
}
