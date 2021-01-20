import torch
import numpy as np

np_y = np.ones([5,3])

ts_y = torch.tensor(np_y)

data = [1,2,3]

ts_data = torch.tensor(data, dtype=torch.float64)
