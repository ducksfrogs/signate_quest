
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()

df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)

df_iris['target'] = iris.target
df_iris.loc[df_iris['target']==0, 'target'] = 'setosa'
df_iris.loc[df_iris['target']==1, 'target'] = 'versicolor'
df_iris.loc[df_iris['target']==2, 'target'] = 'virginica'

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
