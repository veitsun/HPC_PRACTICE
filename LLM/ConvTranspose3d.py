import torch.nn as nn
import torch

# With square kernels and equal stride
m = nn.ConvTranspose3d(16, 33, 3, stride=2)

# non-square kernels and unequal stride and with padding
m = nn.ConvTranspose3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(0, 4, 2))


input = torch.randn(20, 16, 10, 50, 100)
output = m(input)