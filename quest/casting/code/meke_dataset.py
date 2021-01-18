import torch
from torchvision import transforms, datasets
import zipfile

def unzip_dataset(INPATH, OUTPATH):
    with zipfile.ZipFile(INPATH) as zf:
        zf.extractall(OUTPATH)

unzip_dataset(INPATH='./image_data.zip', OUTPATH='./')



data_transforms = {
    'train':transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [.229, 0.224, 0.225]),
    ]),
}

image_datasets = {
    'train': datasets.ImageFolder('./image_data/train', data_transforms['train']),
    'val': datasets.ImageFolder('./image_data/val', data_transforms['val'])
}

image_dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=4, shuffle=True, num_workers=0, drop_last=True),
    'va': torch.utils.data.DataLoader(image_datasets['val'], batch_size=4, shuffle=True, num_workers=0, drop_last=True),
}

for i, (inputs, labels) in enumerate(image_dataloaders):
    print(inputs)
    print(labels)
    if i == 0:
        break
