import os
import zipfile


files = os.listdir('./')

def unzip_dataset(INPATH, OUTPATH):
    with zipfile.ZipFile(INPATH) as zf:
        zf.extractall(OUTPATH)

Unzip_dataset(INPATH='../input/train_data.zip', OUTPATH='./')

tmpdir = '../input/train_tmp'
ok_dir = '../input/train/ok'
ng_dir = '../input/train/ng'

os.mkdir(tmpdir)
os.mkdir(ok_dir)
os.mkdir(ng_dir)


Unzip_dataset(INPATH='../input/train_data.zip', OUTPATH='./')

import glob
import shutil


files = glob.glob(tmpdir)

for file in files:
    if 'ok' in file:
        shutil.copy2(file, ok_dir)
    elif 'def' in file:
        shutil.copy2(file, ng_dir)
    else:
        pass


from sklearn.model_selection import train_test_split

t
