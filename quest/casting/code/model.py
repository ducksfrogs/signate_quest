import os

files = os.listdir('./')

import zipfile

def unzip_dataset(INPATH, OUTPATH):
    with zipfile.ZipFile(INPATH) as zf:
        zf.extractall(OUTPATH)


unzip_dataset(INPATH='./image_data.zip', OUTPATH='./')

ok_image_name_list = os.listdir('./input_data/train/ok')
ng_image_name_list = os.listdir('./input_data/train/ng')

print(len(ok_image_name_list))
print(len(ng_image_name_list))

print(len(set(ok_image_name_list)))
print(len(set(ng_image_name_list)))

from torchvision import transforms

data
