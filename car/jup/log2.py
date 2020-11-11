import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.linear_model import LinearRegression as LR


data = pd.read_csv('data.tsv', sep='\t')
data = data.drop(columns=['id'])

data = data.dropna()

data['displacement_log'] = np.log(data['displacement'])

plt.figure(figsize=(5,5))
plt.scatter(x=data['displacement_log'],y=data['kpl'])

plt.xlabel('displacement_log')
plt.ylabel('kpl')

y = data['kpl']
X = data[['cylinders', 'horsepower', 'acceleration','model_year','origin', 'drive_system','displacement_log']]
X = pd.get_dummies(X)

X_train, X_test, y_train,y_test = train_test_split(X, y, random_state=1)

lr = LR()
lr.fit(X_train, y_train)

y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)

mse_train = MSE(y_train, y_pred_train)
mse_test = MSE(y_test, y_pred_test)

rmse_train = np.sqrt(mse_train)
rmse_test =np.sqrt(mse_test)

print(rmse_train)
print(rmse_test)
