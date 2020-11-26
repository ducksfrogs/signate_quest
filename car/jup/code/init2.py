import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


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


origin = test.pop('origin')
test['USA'] = (origin==1) * 1.0
test['Europe']= (origin==2) * 1.0
test['Japan']= (origin==3) * 1.0
