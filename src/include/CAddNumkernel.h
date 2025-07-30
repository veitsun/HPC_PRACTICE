#pragma once
#include <iostream>

#include "AddNumkernel.cuh"

using namespace std;

#define DX 100

class CAddNumkernel {
 public:
  int *a;
  int *b;
  int *c;

  void setParameter();
  void addNum();
  void show();
  void evolution();
};
