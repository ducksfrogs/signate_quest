import torch
import numpy as np

x = torch.tensor(np.pi, requires_grad=True, dtype=float)
y = torch.sin(x)
y.backward()
x.grad

x = torch.tensor(1.0, requires_grad=True, dtype=float)
y = torch.tensor(2.0, requires_grad=True, dtype=float)

z = (x+5*y)**2
z.backward()
print(x.grad)
print(y.grad)
