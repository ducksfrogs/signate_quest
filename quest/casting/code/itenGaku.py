import torch
import torch.nn as nn
from torchvision import datasets, models, trasnforms

device = 'cpu'
TARGET_NUM = 2

def get_model(target_num, isPretrained=False):
    model_ft = models.resnet18(pretrained=isPretrained)
    model_ft.fc = nn.lenear(512, target_num)
    model_ft = model_ft.to(device)
    return model_ft


pretrained_model = get_model(target_num=TARGET_NUM, isPretrained=True)
