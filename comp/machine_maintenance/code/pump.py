import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix

import librosa

trainX, trainY = train_df[['mean','zc']], train_df['label']
testX, testY = test_df[['mean','zc']], test_df['label']

sc = StandardScaler()
sc.fit(trainX)

trainX = sc.transform(trainX)
testX = sc.transform(testX)


model = OneClassSVM()
model.fit(trainX)

y_pred = model.predict(testX)
pred = np.where(pred==-1, 1, 0)

def create_power_spectral(data):
    N = data.shape[1]
    dt = 10/N
    F = np.abs(np.fft.fft(data)/(N/2))
    fq = np.linspace(0,1/dt, N)
    return F[:, :int(N/2)+1], fq[:int(N/2)+1]

F, freq = create_power_spectral(train)

plt.plot(freq, F[0])

melspec = librosa.feature.melspectrogram(train[0])

librosa.display.specshow(melspec, x_axis='time', y_axis='mel')
melspec_db = librosa.amplitude_to_db(melspecm)

librosa.display.specshow(melspec_db, x_axis='time', y_axis='mel')

train = read_data('pump', 'train', 'normal')
melspec_dbs = []

for i in range(len(train)):
    melspec = librosa.feature.melspectrogram(train[i])
    melspec_db = librosa.amplitude_to_db(melspec).flatten()
    melspec_dbs.append(melspec_db.astype(np.float16))
train_df = pd.DataFrame(melspec_dbs)

for i in range(len(test)):
    melspec =oooo 
