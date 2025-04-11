import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

device = 'cuda'
activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA, ProfilerActivity.XPU]

# 实例化一个简单的 resnet 模型

model = models.resnet18().to(device)
inputs = torch.randn(5, 3, 224, 224).to(device)


# with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
#     with record_function("model_inference"):
#         model(inputs)

with profile(activities=activities) as prof:
    model(inputs)


# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

prof.export_chrome_trace("trace.json")