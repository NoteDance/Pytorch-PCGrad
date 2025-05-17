# Pytorch-PCGrad
PyTorch Implementation of "Gradient Surgery for Multi-Task Learning" using multiprocessing

# Usage
```python
import torch
import torch.nn as nn
import torch.optim as optim
from ppcgrad import PPCGrad

# wrap your favorite optimizer
optimizer = PPCGrad(optim.Adam(net.parameters())) 
losses = [...] # a list of per-task losses
assert len(losses) == num_tasks
optimizer.pc_backward(losses) # calculate the gradient can apply gradient modification
optimizer.step()  # apply gradient step
```
