from sklearn.ensemble import RandomForestClassifier
import librosa

from sklearn.ensemble import RandomForestClassifier

import glob

import numpy as np
import pandas as pd

def read_data(categorie):
    files = sorted(glob.glob(f"../input/{categorie}/*.wav"))
    dataset = []
    for file_name in files:
        y, sr = librosa.load(file_name, sr=None)
        dataset.append(y)

    return np.array(dataset)

normals = read_data('valid_normal')
anomaly = read_data('valid_anomaly')

train = np.concatenate([normals,anomaly])

train_mean = np.sqrt(np.mean(train**2, axis=1))
train_zc = np.sum(librosa.zero_crossings(train), axis=1)

train_df = pd.DataFrame()
train_df['mean'] = train_mean
train_df['zc'] = train_zc
train_df['label'] = np.concatenate([np.zeros(len(normals)), np.ones(len(anomaly))])

tainX, trainy = train_df[['mean', 'zc'], train_df['label']
testX, testy = test_df[['mean', 'zc']], test_df['label']

model = RandomForestClassifier()
model.fit(trainX, trainy)
pred = model.predict(testX)

from sklearn.metrics import confusion_matrix

confusion_matrix(testy, pred)
