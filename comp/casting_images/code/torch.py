import torch
from torchvision import transforms, datasets

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RadomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[.229, ,0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[.229, ,0.224, 0.225])
    ])
}

image_datasets = {
    'train': datasets.ImageFolder("../input/train", data_transforms['train']),
    'val': datasets.ImageFolder("../input/val", data_transforms['val'])
}

image_dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=4, shuffle=True, num_workers=0, drop_last=True),
    'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=4, shuffle=False, num_workers=0, drop_last=True),
    }


from torchvision import datasets, models, transforms

model_ft = models.resnet18(pretrained=False)

print(model_ft)

device = 'cpu'
TARGET_NUM = 10

def get_model(target_num, isPretrained=False):
    model_ft = models.resnet18(pretrained=isPretrained)
    model_ft = model_ft.to(device)
    return model_ft

model = get_model(TARGET_NUM, isPretrained=False)
