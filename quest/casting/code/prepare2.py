import zipfile
import os

def unzip_dataset(PATH):
    with zipfile.ZipFile(PATH) as zf:
        zf.extractall()

unzip_dataset(PATH='./image_data.zip')

ok_image_name_list = os.listdir('./image_data/train/ok')
ng_image_name_list = os.listdir('./image_data/train/ng')

print(len(ok_image_name_list))
print(len(ng_image_name_list))

print(len(set(ok_image_name_list)))
print(len(set(ng_image_name_list)))

from torchvision import transforms, datasets

data_transforms ={
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

image_datasets = {
    'train': datasets.ImageFolder('./image_data/train', data_transforms['train']),
    'val': datasets.ImageFolder("./image_data/val", data_transforms['val'])
}

image_dataloaders = {
 'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=4, num_workers=0, drop_last=True)
 'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=4, num_workers=0, drop_last=True)
}


from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim

model_ft = models.resnet18(pretraind=False)

device = 'cpu'
TARGET_NUM  = 10

def get_model(target_num, isPretrained=False):
    model_ft = models.resnet18(pretraind=isPretrained)
    model_ft.fc = nn.Linear(512, target_num)
    model_ft = model_ft.to(device)
    return model_ft

model = get_model(TARGET_NUM, isPretrained=False)

optimizer = optim.SGD(model_ft.parameters(),lr=0.0001, momentum=0.9)

criterion = nn.CrossEntropyLoss()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
from PIL import Image, ImageFilter

DEVICE = 'cpu'

def train_model(model, criterion, optimizer, num_epochs=5, is_saved=False):
    best_acc = 0.0

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            print("{} phase".format(phase))

            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_correct = 0

            for i,(inputs, labels) in enumerate(image_dataloaders[phase]):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                loss = criterion(outputs, labels)
                print(" loaders:{}".format(i+1), ' loss:{}'.format(loss))

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print("{} loss: {:.4f} Acc: {:4f}".format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                if(is_saved):
                    torch.save(model.state_dict(), './original_model_{}.pth'.format(epoch))
    print("Best val Acc: {:4f}".format(best_acc))


df_test = pd.Dataframe(data=os.listdir('./test_data/'))
df_test = df_test.rename(columns={0: 'filename'})
df_test['target'] = 0

df_test.loc[df_test['filename'].str.contains('ok'), 'target'] = 1

df_test.to_csv("df_test.csv", index=False)


import pandas as pd
import numpy as np


test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class Test_Datasets(Dataset):

    def __init__(self, data_transform):
        self.df = pd.read_csv(('./df_test.csv', names=['filename', 'target']))
        self.data_transform = data_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        file = self.df['filename'][index]
        image = Image.open('./test_data/'+file)
        image = self.data_transform(image)

        return image, file


 test_dataset = Test_Datasets(data_transform=test_transforms)


 test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=0,
                                                drop_last=True)

def  get_model(target_num, isPretrained=False):
    model_ft = models.resnet18(pretraind=isPretrained)
    model_ft.fc = nn.Linear(512, target_num)
    model_ft = model_ft.to(DEVICE)

best_model = get_model(target_num=2)

best_model.load_state_dict(torch.load('./original_model_33.pth', map_location=lambda storage, loc: storage), strict=True)

pred = []

for i, (inputs, labels) in enumerate(test_dataloader):
    inputs = inputs.to(DEVICE)

    best_model.eval()
    outputs = best_model(inputs)

    _, preds = torch.max(outputs, 1)

    pred.append(preds.item())

df_test['pred'] = pred
