import matplotlib.pyplot as plt

import numpy as np
from skimage import io
import glob

files = glob.glob("train_*.jpg")

train_imgs = []

for fname in files:
    img = io.imread(fname)
    train_imgs.append(img)

print(type(train_imgs))
train_imgs = np.array(train_imgs)
print(type(train_imgs), train_imgs.shape)

plt.figure(figsize=(12,5))

for i in range(10):
    plt.subplot(2,5, i+1)
    plt.imshow(train_imgs[i])

plt.show()
