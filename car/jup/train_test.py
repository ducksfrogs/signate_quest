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


data = pd.read_csv('data.tsv', sep='\t')
data = data.drop(columns=['id'])

data = data.dropna()

y = data['kpl']

X = data[['cylinders', 'displacement', 'horsepower', 'acceleration','model_year','origin']]

from sklearn.model_selection import train_test_split

X_train,X_test,y_train, y_test = train_test_split(X, y, random_state=32)


from sklearn.linear_model import LinearRegression as LR

lr = LR()

lr.fit(X_train, y_train)

y_pred_train = lr.predict(X_train)
print(y_pred_train)

from sklearn.metrics import mean_squared_error as MSE

mse_train = MSE(y_train, y_pred_train)

rmse_train = np.sqrt(mse_train)

print(rmse_train)

y_pred_test = lr.predict(X_test)

print(y_pred_test)

mse_test = MSE(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)

print(rmse_train)
print(rmse_test)

plt.figure(figsize=(5,5))
plt.scatter(y_test, y_pred_test)

test_min = np.min(y_test)
test_max = np.max(y_test)
print(test_min, test_max)

pred_min = np.min(y_pred_test)
pred_max = np.max(y_pred_test)
print(pred_min, pred_max)

min_value = np.minimum(test_min, pred_min)
max_value = np.maximum(test_max, pred_max)
print(min_value, max_value)

plt.xlim([min_value, max_value])
plt.ylim([min_value, max_value])

plt.plot([min_value, max_value],[min_value,max_value])

plt.xlabel('予測値')
plt.ylabel('実測値')
