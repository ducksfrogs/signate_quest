import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4,12)
        self.fc2 = nn.Linear(12,8)
        self.fc3 = nn.Linear(8,3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)

#second net
class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet,self).__init__()
        self.fc1 = nn.Linear(4,12)
        self.fc2 = nn.Linear(12, 8)
        self.fc3 = nn.Linear(8, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


#third net

class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4,12)
        self.fc2 = nn.Linear(12, 8)
        self.fc3 = nn.Linear(8,4)
        self.fc4 = nn.Linear(4, 3)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

net = IrisNet()
print(net(torch.tensor([[5.1000, 3.5000, 1.4000, 0.2000]])))

#Kfold

from sklearn.model_selection import KFold

kf = KFold(n_splits=3, shuffle=True, random_state=1993)
for fold, (train_index, test_index) in enumerate(kf.split(iris.data, iris.target)):
    print("Flold {}".format(fold), "len(train_index):{}".format(len(train_index)), "len test_index:{}".format(len(test_index)))
