import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn import datasets
from sklearn.model_selection import KFold

class IrisNet(nn.Module):

    def __init__(self):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4,12)
        self.fc2 = nn.Linear(12,8)
        self.fc3 = nn.Linear(8,3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


FOLD = 3
IS_SHUFFLE = True
SEED = 1234
EPOCH = 5
LEARNING_RATE = 0.01

iris = datasets.load_iris()

kf = KFold(n_splits=FOLD,shuffle=IS_SHUFFLE, random_state=SEED)

for fold, (train_index, test_index) in enumerate(kf.split(iris.data, iris.target)):
    print("Fold: {}".format(fold), "len(train_index): {}".format(len(train_index)), "len(test_index)".format(len(test_index)))
    X_train = iris.data[train_index]
    X_test = iris.data[test_index]
    y_train = iris.target[train_index]
    y_test = iris.target[test_index]

    x = torch.tensor(X_train, dtype=torch.float)
    y = torch.tensor(y_train)
    net = IrisNet()

    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE)

    criterion = nn.CrossEntropyLoss()

    for i in range(EPOCH):
        optimizer.zero_grad()
        output = net(x)

        loss = criterion(output, y)
        loss.backward()

        optimizer.step()
        print()
