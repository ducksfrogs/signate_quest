import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import train_test_split
from sklearn.metrics import mean_squared_error as MSE


select_columns = ['id', 'y','capa','week','home','away','stage', 'tv_num', 'month','weather']

dummy_data_all = pd.get_dummies(data[select_columns],drop_first=True)

dummy_data = dummy_data_all[dummy_data_all['id'] <=16804]
dumm_data_target = dummy_data_all[dummy_data_all['id']>=16805]
y_1 = dumm_data['y']
dumm_data = dumm_data.drop(['id', 'y'])

X_train, X_test, y_train, y_test =  train_test_split(dumm_data, y_1, random_state=1234)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_train = lr.predict(X_train)
mse = MSE(y_train, y_pred_train)

rmse_train = np.sqrt(mse)
y_pred_test = lr.predict(X_test)

X_train['pred'] = y_pred_train
X_train['res'] = y_train - X_train['pred']
X_test['pred'] = y_pred_test
X_test['res'] = y_test - X_test['pred']

tmp = pd.concat([X_train,X_test], axis=0)

data = pd.concat([data, tmp[['pred','res']]], axis=1)

data.sort(by='res', ascending=True)

y_2 = dummy_data_target['y']
dummy_data_target = dumm_data_target.drop(['id','y'], axis=1)
target_pred = lr.predict(dumm_data_target)
