import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.linear_model import LinearRegression as LR


data = pd.read_csv('data.tsv', sep='\t')
data = data.drop(columns=['id'])

data = data.dropna()

y = data['kpl']

X = data[['cylinders', 'displacement', 'horsepower', 'acceleration','model_year','origin']]
X = pd.get_dummies(X)

X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=1)


lr = LR()
lr.fit(X_train, y_train)

y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)

mse_train = MSE(y_train, y_pred_train)
mse_test = MSE(y_test, y_pred_test)

rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)

print(y_pred_train)
print(rmse_train)
print(rmse_test)

#plt.figure(figsize=(5,5))
#plt.scatter(y_test, y_pred_test)
#
#test_min = np.min(y_test)
#test_max = np.max(y_test)
#print(test_min, test_max)
#
#pred_min = np.min(y_pred_test)
#pred_max = np.max(y_pred_test)
#print(pred_min, pred_max)
#
#min_value = np.minimum(test_min, pred_min)
#max_value = np.maximum(test_max, pred_max)
#print(min_value, max_value)
#
#plt.xlim([min_value, max_value])
#plt.ylim([min_value, max_value])
#
#plt.plot([min_value, max_value],[min_value,max_value])
#
#plt.xlabel('予測値')
#plt.ylabel('実測値')
