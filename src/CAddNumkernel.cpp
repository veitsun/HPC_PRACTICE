#include "../include/CAddNumkernel.h"

void CAddNumkernel::setParameter() {
  cudaMallocManaged(&a, sizeof(int) * DX);
  cudaMallocManaged(&b, sizeof(int) * DX);
  cudaMallocManaged(&c, sizeof(int) * DX);

  for (int f = 0; f < DX; f++) {
    a[f] = f;
    b[f] = f + 1;
  }
}

// 这个成员函数调用了一个函数，这个函数里面调用了内核函数
void CAddNumkernel::addNum() { AddKernel(a, b, c, DX); }

void CAddNumkernel::show() {
  cout << " a     b    c" << endl;

  for (int f = 0; f < DX; f++) {
    cout << a[f] << " + " << b[f] << "  = " << c[f] << endl;
  }
}

void CAddNumkernel::evolution() {
  setParameter();
  addNum();
  show();
}