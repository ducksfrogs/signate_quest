import zipfile
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, datasets


def unzip_dataset(INPATH, OUTPATH):
    with zipfile.ZipFile(INPATH) as zf:
        zf.extractall(OUTPATH)

unzip_dataset(INPATH='./augment_data.zip', OUTPATH='./')


def show_augment(dataset):
    for i in range(0,6):
        ax = plt.subplot(2,3,i+1)
        plt.tight_layout()
        ax.set_title(str(i))
        plt.imshow(dataset[i][0])
    plt.show()

transform = transforms.Compose([
        transforms.RondomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
])
aug_dataset = datasets.ImageFolder(root='./augment_data/', transform=transform)
show_augment(aug_dataset)
