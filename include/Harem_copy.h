#ifndef HAREM_COPY_H
#define HAREM_COPY_H

class Harem_copy {
private:
  int age;
  int height;
  int weight;
  bool beauty;

public:
  Harem_copy(int age, int height, int weight);
  bool check_beauty();
  void show();
  ~Harem_copy();
};

#endif