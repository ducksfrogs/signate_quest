import librosa

file_path = '../input/train_normal/000.wav'

y, sr = librosa.load(file_path, sr=None)

import glob
import numpy as np

category = 'train_normal'

def read_data(category):
    files = sorted(glob.glob(f"../input/{category}/*.wav"))
    dataset = []
    for file_name in files:
        y, sr = librosa.load(file_name, sr=None)
        dataset.append(y)
    return np.array(dataset)

s = read_data(category)

import matplotlib.pyplot as plt

normal, sr = librosa.load(file_path, sr=None)
plt.plot(normal)

anomal_path = '../input/valid_anormaly'

anormaly, sr = librosa.load(file_path, sr=None)
plt.plot(anormaly, color='orange')

normal_mean = np.sqrt(np.mean(s**2, axis=1))

anormaly = read_data('valid_anormaly')

anormaly_mean = np.sqrt(np.mean(anormaly**2, axis=1))

plt.hist(normal_mean, alpha=0.5)
plt.hist(anormaly_mean, alpha=0.5)
