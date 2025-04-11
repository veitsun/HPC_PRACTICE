import torch.nn as nn
import torch

# With square kernels and equal stride
conv_transpose = nn.ConvTranspose3d(16, 33, 3, stride=2)

# non-square kernels and unequal stride and with padding
m = nn.ConvTranspose3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(0, 4, 2))

# 假设输入的张量形状为 (batch_size=20, channels=16, depth=10, height=50, width=100)
# 有 20 个样本，每个样本有 16 个通道，每个通道的形状为 10x50x100
input_tensor = torch.randn(20, 16, 10, 50, 100)

output_tensor = conv_transpose(input_tensor)

print(output_tensor.shape)  # torch.Size([20, 33, 21, 101, 201])
# 输出的张量尺寸被放大了，通道数从原来的 16 变成了 33，深度从 10 变成了 21，高度从 50 变成了 101，宽度从 100 变成了 201