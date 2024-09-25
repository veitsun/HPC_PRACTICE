#include "../include/CInitialData.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>
void CInitialData::initialData(float *ip, int size) {
  // generate different seed for random number
  time_t t;
  srand((unsigned)time(&t));

  for (int i = 0; i < size; i++) {
    ip[i] = (float)(rand() & 0xFF) / 10.0f;
  }

  return;
}