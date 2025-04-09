import torch

import triton
import triton.language as tlp

@triton.jit  # 每一个 triton 内核都需要以一个装饰器开始，这是告诉 triton 这段代码将要被编译
def add_kernel(
  x_ptr,
  y_ptr,
  output_ptr,   # 输出向量的指针
  n_elements,   # 向量中元素的数量
  BLOCK_SIZE: tlp.constexpr, # 每个程序应该处理元素的数量
):
  pid = tlp.program_id(axis=0) # 获取程序 ID
  block_start = pid * BLOCK_SIZE
  offsets = block_start + tlp.arange(0, BLOCK_SIZE)
  mask = offsets < n_elements
  x = tlp.load(x_ptr + offsets, mask=mask)
  y = tlp.load(y_ptr + offsets, mask=mask)
  output = x + y
  tlp.store(output_ptr + offsets, output, mask=mask) # 将x + y 的结果存放到 output_ptr 向量中

# 写了一个 wrapper 包装器
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  # 获取输入向量的大小
  n_elements = x.numel()
  # 创建一个输出向量
  output = torch.empty_like(x)
  # 获取每个程序应该处理的元素的数量
  BLOCK_SIZE = 1024
  # 调用内核
  grid = lambda opt: (triton.cdiv(n_elements, opt["BLOCK_SIZE"]),)
  add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
  return output

if __name__ == "__main__":
  # 创建两个随机向量
  x = torch.randn(1024 * 1024, device='cuda')
  y = torch.randn(1024 * 1024, device='cuda')
  # 调用 add 函数
  output = add(x, y)
  # 打印输出
  print(output)
  # 验证输出是否正确
  assert torch.allclose(output, x + y), "Output is incorrect"
  print("Output is correct")
  # 打印输出
  print("Output: ", output)
  # 打印输出的形状
  print("Output shape: ", output.shape)
  # 打印输出的类型
  print("Output type: ", output.dtype)
  # 打印输出的设备
  print("Output device: ", output.device)
  # 打印输出的大小
  print("Output size: ", output.numel())
  # 打印输出的内存占用
  print("Output memory size: ", output.element_size() * output.numel())
  # 打印输出的内存占用
  print("Output memory size (MB): ", output.element_size() * output.numel() / (1024 ** 2))