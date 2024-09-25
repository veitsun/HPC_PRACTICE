#include "../include/Harem_copy.h"
#include <iostream>

Harem_copy::Harem_copy(int age, int height, int weight) {
  this->height = height;
  this->weight = weight;
  this->age = age;
}

bool Harem_copy::check_beauty() {
  if (this->age > 16 && this->age < 25) {
    if (height > 150 && height < 170) {
      if (weight > 50 && weight < 60) {
        // std::cout << "beautiful !!!" << std::endl;
        return true;
      }
    }
  }
  return false;
}

void Harem_copy::show() {
  std::cout << "age : " << age << " height : " << height
            << " weight : " << weight << std::endl;
}

Harem_copy::~Harem_copy() {}