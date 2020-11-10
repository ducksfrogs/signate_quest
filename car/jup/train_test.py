import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#data = pd.read_csv('data.tsv', sep='\t')
#data = data.drop(columns=['id'])
#
#data = data.dropna()
#
#y = data['kpl']
#
#X = data[['cylinders', 'displacement', 'horsepower', 'acceleration','model_year','origin']]
#

#second

#from sklearn.model_selection import train_test_split
#
#X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=32)
#
#X_train_line = X_train.shape[0]
#X_test_line = X_test.shape[0]
#
#print(X_train_line)
#print(X_test_line)

#third

actual = np.array([4,6,8], dtype='float32')
pred = np.array([2,8,10], dtype='float32')

a = actual - pred
print('a', a)

b = np.power(a,2)
print('b', b)

c = np.sum(b)
print('c', c)

d = c / len(a)
print('d', d)

e = np.sqrt(d)
print(e)

from sklearn.metrics import mean_squared_error as MSE

# 変数の準備
actual = [3,4,6,2,4,6,1]
pred = [4,2,6,5,3,2,3]

mse = mse(actual, pred)
print(mse)

rmse = np.sqrt(mse)
print(rmse)


#modeling the first
