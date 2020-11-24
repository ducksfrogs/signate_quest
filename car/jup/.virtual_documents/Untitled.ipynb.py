import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('../data/train.csv', index_col=0)
test = pd.read_csv('../data/test.csv', index_col=0)


df.head()


from sklearn.model_selection import train_test_split



test


df.corr()


trainX = df[[ 'cylinders', 'displacement']]





Y = df['mpg']


from sklearn.linear_model import LinearRegression


trainX.info()





clf = LinearRegression()


clf.fit(trainX, Y)





train


train.corr()





testX = test[['cylinders', 'displacement']]


testX.info()


pred = clf.predict(testX)


sample = pd.read_csv('../data/sample.csv',header=None)


sample.head()


sample[1] = pred


sample.head()


sample.to_csv('submit2.csv', index=None, header=None)



