import os
import zipfile
from sklearn.model_selection import train_test_split

def unzip_dataset(INPATH, OUTPATH):
    with zipfile.ZipFile(INPATH) as zf:
        zf.extractall(OUTPATH)


tmpdir = '../input/train_tmp'

ok_dir = '../input/train/ok'
ng_dir = '../input/train/ng'

val_ok_dir = '../input/val/ok'
val_ng_dir = '../input/val/ng'

os.mkdir(tmpdir)
unzip_dataset(INPATH='../input/train_data.zip', OUTPATH='./')


files = glob.glob(tmpdir+'/*/*')

train_files, test_files = train_test_split(files)


os.mkdir(ok_dir)
os.mkdir(ng_dir)

os.mkdir(val_ok_dir)
os.mkdir(val_ng_dir)


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
