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
