import librosa
import numpy as np
import matplotlib.pyplot as plt


file_path =  '../input/normal/00.wav'

y, sr = librosa.load(file_path, sr=None)

import glob

category = 'valve'
train_test = 'train'

def read_data(category, train_test, label):
    files = sorted(glob.glob(f'dataset/{category}/{train_test}/{label}/*.wav'))
    dataset = []
    for file_name in files:
        y, sr = librosa.load(file_name, sr=None)
        dataset.append(y)

    return np.array(dataset)

valve = read_data('valve', 'train', 'normal')

normal, sr = librosa.load(file_path, sr=None)
plt.plot(normal)

anormal, sr = librosa.load('/dataset/valve/train/normal/00.wav', sr=None)
plt.plot(anormal, color='orange')
