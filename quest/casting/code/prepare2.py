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
