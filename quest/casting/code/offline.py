import torch
import torch.nn as nn
from torchvision import datasets, models, transforms


device = 'cpu'
TARGET_NUM = 2


def get_model(target_num, isPretrained=False):

    if(isPretrained):
        model_ft = models.resnet18(pretrained=False)
        model_ft.load_state_dict(torch.load('./resnet18-5c106cde.pth', map_location=lambda storage, loc: storage), strict=True)
    else:
        model_ft = models.resnet18(pretrained=False)

    model_ft.fc = nn.Linear(512, target_num)
    model_ft = model_ft.to(device)

pretrained_model = get_model(target_num=2, isPretrained=True)
