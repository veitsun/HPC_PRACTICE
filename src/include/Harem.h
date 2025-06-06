#ifndef HAREM_H
#define HAREM_H

class Harem {
private:
  int age;
  int height;
  int weight;
  bool beauty;

public:
  Harem(int age, int height, int weight);
  bool check_beauty();
  void show();
  ~Harem();
};

#endif