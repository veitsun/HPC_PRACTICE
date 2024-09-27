#include "../include/CTest.h"

void CTest::setParameter() {
  cudaMallocManaged(&a, sizeof(int) * DX);
  cudaMallocManaged(&b, sizeof(int) * DX);
  cudaMallocManaged(&c, sizeof(int) * DX);

  for (int f = 0; f < DX; f++) {
    a[f] = f;
    b[f] = f + 1;
  }
}

void CTest::addNum() { AddKernel(a, b, c, DX); }

void CTest::show() {
  cout << " a     b    c" << endl;

  for (int f = 0; f < DX; f++) {
    cout << a[f] << " + " << b[f] << "  = " << c[f] << endl;
  }
}

void CTest::evolution() {
  setParameter();
  addNum();
  show();
}