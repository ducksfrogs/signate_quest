import zipfile
import torch
import matplotlib.pyplot as plt
import albumentations
from PIL import Image
from torchvision import transforms, datasets
import numpy as np



def unzip_dataset(INPATH, OUTPATH):
    with zipfile.ZipFile(INPATH) as zf:
        zf.extractall(OUTPATH)

unzip_dataset(INPATH='./augment_data.zip', OUTPATH='./')

#transform = transforms.Compose([
#        transforms.RondomHorizontalFlip(p=0.5),
#        transforms.RandomVerticalFlip(p=0.5),
#])
#aug_dataset = datasets.ImageFolder(root='./augment_data/', transform=transform)
#show_augment(aug_dataset)

albu_transforms = albumentations.Compose([
                    albumentations.RandomGridShuffle(grid=(1,1),p=1.0)
])

def albumentations_transform(image, transform=albu_transforms):
    image_np = np.array(image)
    augmented = transform(image=image_np)
    image = Image.fromarray(augmented['image'])
    return image

data_transform = transforms.Compose([
                    transforms.Lambda(albumentations_transform),
])

dataset_augmentated = datasets.ImageFolder(root='./augment_data/', transform=data_transform)

def show_augment(dataset):
    for i in range(0,6):
        ax = plt.subplot(2,3,i+1)
        plt.tight_layout()
        ax.set_title(str(i))
        plt.imshow(dataset[i][0])
    plt.show()

show_augment(dataset_augmentated)
