import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import pandas as pd

import glob

files = (glob.glob("train_*.jpg"))

train_imgs = []
for fname in files:
    img = io.imread(fname)
    train_imgs.append(img)
train_imgs = np.array(train_imgs)

plt.figure(figsize=(12,5))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(train_imgs[i])

labels = []

for fname in files:
    labels.append(fname.split('_')[1])

labels_dummy = pd.get_dummies(labels)


X_tr, X_val = np.split(train_imgs, [8])
y_tr, y_val = np.split(labels_dummy, [8])

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
