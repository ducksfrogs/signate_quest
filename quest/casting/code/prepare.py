import glob
import os
import re
import shutil

os.makedirs(dst_dir, exist_ok=True)

src_dir = 'images'
dst_dir = 'output'

ok_path = "./ok"
ng_path = "./ng"

data_file = glob.glob(os.path.join(src_dir, '*.jpeg'))

ok_path = './input/'
for img in data_file:
    filename = os.path.basename(img)
    if 'ok' in filename:
        dst = os.path.join(ok)

train_val = ['', '']



for i in train_val:
    tmp_path = './input/{}'.format(i)
    data_file = glob.glob(os.path.join(tmp_path, '*.jpeg'))
    for img in data_file:
        filename = os.path.basename(img)
        if 'os' in filename:
            dst = os.path.join(ok_path, filename)
            shutil.copy2(img, dst)
        else:
            dst = os.path.join(ng_path, filename)
            shutil.copy2(img, dst)

            
