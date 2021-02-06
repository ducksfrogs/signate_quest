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

from torchvision import transforms

data_transform ={
    'train': transforms.Compose([
        transforms.Resize(256)
    ])
}
