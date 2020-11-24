import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('../data/train.csv', index_col=0, na_values='?')
test = pd.read_csv('../data/test.csv', index_col=0, na_values='?')


df.isnull().any()


df.corr()


df.columns


df = df[['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
       'acceleration', 'model year', 'origin']]





sns.pairplot(df[['mpg','cylinders','displacement','weight']], diag_kind='kde')


from sklearn.preprocessing import MinMaxScaler


df.info()


df.describe()


origin = df.pop('origin')


df['USA'] = (origin == 1) *1.0
df['Europe'] = (origin == 2) * 1.0
df['Japan'] = (origin == 3) * 1.0
df.tail()


test


df.corr()


trainX = df[[ 'cylinders', 'displacement']]





Y = df['mpg']


from sklearn.linear_model import LinearRegression


trainX.info()





clf = LinearRegression()


clf.fit(trainX, Y)


testX = d


train


train.corr()





testX = test[['cylinders', 'displacement', 'weight','acceleration', 'model year', 'origin']]


testX.info()


pred = clf.predict(testX)


sample = pd.read_csv('../data/sample.csv',header=None)


sample.head()


sample[1] = pred


sample.head()


sample.to_csv('submit.csv', index=None, header=None)



