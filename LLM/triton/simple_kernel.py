

import triton
import triton.language as tl
# Triton 是一个用于编写高性能 GPU 代码的 Python 库
# 它允许用户使用 Python 语言编写 GPU 内核，并提供了许多用于优化和调试的工具
# Triton 的设计目标是使 GPU 编程更容易，并提供更好的性能
# Triton 的主要特点包括
# - 简单易用的 API：Triton 提供了一个简单易用的 API，使得编写 GPU 内核变得更加容易
# - 高性能：Triton 生成的代码可以与手写的 CUDA 代码相媲美
# - 支持多种硬件架构：Triton 支持多种硬件架构，包括 NVIDIA 和 AMD 的 GPU
# - 支持多种编程语言：Triton 支持多种编程语言，包括 Python 和 C++

i = tl.constexpr(0)

@triton.jit
def simple_kernel(
  BLOCK_SIZE: tl.constexpr,
  n_elements,
  ):
  pid = tl.program_id(axis=0) 
  # i += 1 全局变量 i ， 可以访问呢它，但是不能修改
  block_start = pid * BLOCK_SIZE
  offsets = block_start + tl.arange(0, BLOCK_SIZE)
  mask = offsets < n_elements
  print("Hello from program id", pid) # print 不会在 GPU 上工作



simple_kernel[(1,)](BLOCK_SIZE=16, n_elements=16)