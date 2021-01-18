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
