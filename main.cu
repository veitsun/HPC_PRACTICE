#include "include/Harem.h"
#include "include/Harem_copy.h"
#include <cstdio>
#include <cstring>
#include <iostream>
using namespace std;

int main() {
  Harem *girl = new Harem(18, 165, 105);
  bool check = girl->check_beauty();
  if (check)
    cout << "nice" << endl;
  else
    cout << "sad" << endl;
  girl->show();

  Harem_copy *girl_1 = new Harem_copy(18, 165, 105);
  bool check_copy = girl_1->check_beauty();
  if (check_copy)
    cout << "nice girl" << endl;
  else
    cout << "sad girl" << endl;
  girl_1->show();

  return 0;
}
