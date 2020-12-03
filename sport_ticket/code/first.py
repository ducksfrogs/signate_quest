import pandas as pd

data = pd.read_csv('data.csv')
data_add = pd.read_csv("data_add.csv")
data_new = pd.concat([data, data_add])

condition = pd.read_csv('condition.csv')
condition_add = pd.read_csv('condition_add.csv')
condition_new = pd.concat(condition, condition_add)

tmp = data_new.merge(condition_new,on='id',how='out' )

tmp = pd.merge(data_new, condition_new, on='id', how='left')

stadium = pd.read_csv('stadium.csv')

data_all = pd.merge(tmp, stadium, left_on='stadium', right_on='name', how='left')

print(tmp.shape[1])
print(stadium.shape[1])
print(data_all.shape[1])

print(data_all.head(10))

print(data_all.columns)
data_all = data_all.drop(columns=['name'])

print(data_all.columns)

data_all.to_csv('data_all.csv', index=None)

data = pd.read_csv('data_all.csv')
