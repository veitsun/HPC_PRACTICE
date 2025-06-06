#include "CTransPoseSolution.h"

// 后继
int CTransPoseSolution::getNext(int i, int m, int n) {
  return (i % n) * m + i / n;
}

// 前驱
int CTransPoseSolution::getPre(int i, int m, int n) {
  return (i % m) * n + i / m;
}
// 处理下标为i起点的环
void CTransPoseSolution::movedata(float *mtx, int i, int m, int n) {
  int temp = mtx[i]; // 暂存
  int cur = i;       // 当前下标
  int pre = getPre(cur, m, n);
  while (pre != i) {
    mtx[cur] = mtx[pre];
    cur = pre;
    pre = getPre(cur, m, n);
  }
  mtx[cur] = temp;
}

// 转置，即循环处理所有环
void CTransPoseSolution::transpose(float *mtx, int m, int n) {
  for (int i = 0; i < m * n; i++) {
    int next = getNext(i, m, n);
    while (next > i) { // 若存在后继小于i说明重复
      next = getNext(next, m, n);
    }
    if (next == i) { // 处理当前环
      movedata(mtx, i, m, n);
    }
  }
}