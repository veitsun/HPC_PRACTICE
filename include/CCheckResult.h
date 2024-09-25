#ifndef checkResult_H
#define checkResult_H

class CCheckResult {
public:
  void checkResult(float *hostRef, float *gpuRef, const int N);
};

#endif