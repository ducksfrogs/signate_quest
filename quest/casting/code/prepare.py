import glob
import os
import re
import shutil

src_dir = 'images'
dst_dir = 'output'

os.makedirs(dst_dir, exist_ok=True)

for img in glob.glob(os.path.join(src_dir, '*.jpeg')):
    filename = os.path.basename(img)

    if 'ok' in filename:
        dst_path = os.path.join(dst_dir, filename)
        shutil.copy(img, dst_path)

    else:
