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

test
