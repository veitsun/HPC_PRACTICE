#ifndef CTRANSPOSESOLUTION_H
#define CTRANSPOSESOLUTION_H

class CTransPoseSolution {
public:
  int getNext(int i, int m, int n);
  int getPre(int i, int m, int n);
  void movedata(float *mtx, int i, int m, int n);
  void transpose(float *mtx, int m, int n);
};

#endif
