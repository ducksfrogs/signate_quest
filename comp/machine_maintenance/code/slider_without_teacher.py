import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import librosa

import glob

def read_data(category):
    files = sorted(glob.glob(f"../input/{category}/*.wav"))
    dataset = []
    for file_name in files:
        y, sr = librosa.load_data(file_name, sr=None)
        dataset.append(y)
    return np.array(dataset)

train = read_data('train')
train_df = pd.DataFrame()
train_df['mean'] = np.sqrt(np.mean(train**2, axis=1))
train_df['zc'] = np.sum(librosa.zero_crossings(train), axis=1)

test_normal = read_data('test')
test_anomal = read_data('test_anomal')

test = np.concatenate([test_normal, test_anomal])

test_df = pd.DataFrame()
test_df['mean'] = np.sqrt(np.mean(test**2, axis=1))
test_df['zc'] = np.sum(librosa.zero_crossings(test), axis=1)
test_df['label'] = np.concatenate([np.zeros(len(test_normal)), np.ones(len(test_anomal))])

train_x, train_y = train_df[['mean','zc']], train_df['label']
test_x, test_y = test_df[['mean','zc']], test_df['label']

from sklearn import preprocessing

sc = preprocessing.StandardScaler()
sc.fit(train_x)

train_x = sc.transform(train_x)
test_x = sc.transform(test_x)

from sklearn.svm import OneClassSVM

model = OneClassSVM()
model.fit(train_x)
pred = model.predict(test_x)

pred = np.where(pred==-1, 1, 0)
from sklearn.metrics import confusion_matrix

confusion_matrix(test_y, pred)
