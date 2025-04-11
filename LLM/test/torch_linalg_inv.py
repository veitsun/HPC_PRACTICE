import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity


device = 'cuda'
activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

A = torch.randn(1280, 1280).to(device)

with profile(activities=activities, record_shapes=True) as prof:
  with record_function("matrix_inverse"):
    Ainv = torch.linalg.inv(A).to(device)
  with record_function("matrix dist"):
    torch.dist(A @ Ainv, torch.eye(1280, device=device))
    
prof.export_chrome_trace("torch_linalg_inv.json")

A = torch.randn(2, 3, 4, 4)  # Batch of matrices
Ainv = torch.linalg.inv(A)
print(torch.dist(A @ Ainv, torch.eye(4)))

A = torch.randn(4, 4, dtype=torch.complex128)  # Complex matrix
Ainv = torch.linalg.inv(A)
print(torch.dist(A @ Ainv, torch.eye(4)))


