#include "include/Num.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>

const int NUM_RANDOM_NUMBERS = nx * ny;      // 生成随机数的数量
const char *FILENAME = "random_numbers.txt"; // 文件名
int main() {
  // generate different seed for random number
  time_t t;
  srand((unsigned)time(&t));
  std::ofstream outputFile(FILENAME);
  if (!outputFile.is_open()) {
    std::cerr << "无法打开文件 " << FILENAME << std::endl;
    return 1;
  }
  for (int i = 0; i < NUM_RANDOM_NUMBERS; i++) {
    float randomNumber = (float)(rand() & 0xFF) / 10.0f;
    outputFile << randomNumber << " ";
  }
  outputFile.close();
  std::cout << "随机数已成功写入文件 " << FILENAME << std::endl;

  return 0;
}
