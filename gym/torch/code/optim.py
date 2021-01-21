import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn import datasets


class IrsiNet(nn.Module):

    def __init__(self):
        super(IrsiNet, self).__init__()
        self.fc1 = nn.Linear(4,12)
        self.fc2 = nn.Linear(12, 8)
        self.fc3 = nn.Linear(8,4)
        self.fc4 = nn.Linear(4,3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


iris = datasets.load_iris()
X_train = iris.data
y_train = iris.target
x = torch.tensor(X_train, dtype=torch.float)
y = torch.tensor(y_train)

net = IrsiNet()

FOLD = 3
IS_SHUFFLE =True
SEED = 1234
EPOCH = 10
LEARNING_RATE = 0.01


optimizer = optim.SGD(net.parameters(),lr=0.01)
criterion = nn.CrossEntropyLoss()

for i in range(EPOCH):
    optimizer.zero_grad()
    output =  net(x)

    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print("epoch: {}".format(i)+ "loss: {:10f}".format(loss))
