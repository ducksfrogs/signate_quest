import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from  tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


df = pd.read_csv('../data/train.csv', index_col=0, na_values='?')
test = pd.read_csv('../data/test.csv', index_col=0, na_values='?')

mean = np.mean(df['horsepower'])

df['horsepower'].fillna(mean, inplace=True)


origin = df.pop('origin')
df['USA'] = (origin==1) * 1.0
df['Europe']= (origin==2) * 1.0
df['Japan']= (origin==3) * 1.0

train = df[['cylinders',
             'displacement',
             'horsepower',
             'weight',
             'acceleration',
             'model year',
             'USA',
             'Europe',
             'Japan']]

Y = df[['mpg']]

trainX, testX, trainY, testY = train_test_split(train, Y)


scores = []
reg = RandomForestRegressor()

est_range_exp = [1e-1, 1, 1e2, 1e5]
RFR_grid = {"n_estimators": est_range_exp,
                 "max_depth":[i for i in range(1,5)],
                 "random_state": [i for i in range(0,10)]}

#for model, param in tqdm(RFR_grid.items()):
#    reg = GridSearchCV(model, param)
#    reg.fit(trainX, trainY)
#    score = reg.score(testX, testY)
#    scores.append(score)



gs = GridSearchCV(reg, RFR_grid, n_jobs=-1, verbose=2, return_train_score=True)

gs.fit(trainX, trainY)

reg = RandomForestRegressor(max_depth=7, n_estimators=1, random_state=6)
reg.fit(trainX, trainY)
reg.score(testX, testY)

testX = test[['cylinders',
            'displacement',
            'horsepower',
            'weight',
            'acceleration','model year', 'origin']]


origin = testX.pop('origin')
testX['USA'] = (origin==1) * 1.0
testX['Europe']= (origin==2) * 1.0
testX['Japan']= (origin==3) * 1.0

testX['horsepower'].fillna(mean, inplace=True)

pred = reg.predict(testX)

sample = pd.read_csv("../data/", header=None)

sample[1] = pred

sample.to_csv('submit.csv', index=None, header=None)
