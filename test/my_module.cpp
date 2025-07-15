#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // 用于 STL 容器支持

int add(int i, int j) { return i + j; }

// pybind11 绑定代码
PYBIND11_MODULE(my_module, m) {
  m.doc() = "pybind11 example plugin";
  // 绑定函数，设置模块文档字符串
  m.def("add", &add, "A function which adds two numbers");
}