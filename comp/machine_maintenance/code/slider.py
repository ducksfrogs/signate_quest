from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import glob


def read_data(category, train_test, label):
    files = sorted(glob.glob(f"../input/{category}/{train_test}/{label}/*.wav"))
    dataset = []
    for file_name in files:
        y, sr = librosa.load_data(file_name, sr=None)
        dataset.append(y)
    return np.array(dataset)


train_normal = read_data('slider', 'train', 'normal')
train_anomaly = read_data('slider', 'train', 'anomaly')

train = np.concatenate([train_normal, train_anomaly])
train_df = pd.DataFrame()

train_df['mean'] = np.sqrt(np.mean(train**2, axis=1))
train_df['zc'] = np.sum(librosa.zero_crossings(train), axis=1)
train_df['label'] = np.concatenate([np.zeros(len(train_normal)),np.ones(len(train_anomaly))])

test_normal = read_data('slider', 'test', 'normal')
test_anomaly = read_data('slider', 'test', 'anomaly')

test = np.concatenate([test_normal, test_anomaly])
test_df = pd.DataFrame()
test_df['mean'] = np.sqrt(np.mean(test**2, axis=1))
test_df['zc'] = np.sum(librosa.zero_crossings(test), axis=1)
test_df['label'] = np.concatenate([np.zeros(len(test_normal)), np.ones(len(test_anomaly))])


 train_x, train_y = train_df[['mean','zc']], train_df]['label']
 test_x, test_y = test_df[['mean','zc']], test_df['label']

 model.fit(train_x, train_y)

 pred = model.predict(test_x)

 confusion_matrix(test_y, pred)
