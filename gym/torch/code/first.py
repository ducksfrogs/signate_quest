import torch
ts_sample = torch.tensor([[1,2,3],[-4,-5,-6]],dtype=float)

print(torch.abs(ts_sample))

print(torch.std(ts_sample))

x = torch.tensor(1.0, requires_grad=True, dtype=float)
y = torch.tensor(2.0,requires_grad=True, dtype=float)
z=(5*x+y)**3

z.backward()
print(x.grad)
print(y.grad)
