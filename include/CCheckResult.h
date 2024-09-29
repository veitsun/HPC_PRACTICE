#ifndef checkResult_H
#define checkResult_H

class CCheckResult {
public:
  void checkResult(float *hostRef, float *gpuRef, const int N);
  void compare(float *hostC, float *serialC, int M, int N);
};

#endif