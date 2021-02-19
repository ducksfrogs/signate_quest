from torchvision import datasets, models, transforms

model_ft = models.resnet18(pretrained=False)

device = 'cpu'
TARGET_NUM = 10

import torch.nn as nn
import torch.optim as optim


def get_model(target_num, isPretrained=False):

    model_ft = models.resnet18(pretrained=isPretrained)
    model_ft.fc = nn.Linear(512, target_num)
    model_ft = model_ft.to(device)

model = get_model(TARGET_NUM, isPretrained=False)

optimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

criterion = nn.CrossEntropyLoss()
