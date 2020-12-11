import pandas as pd

data = pd.read_csv('train_B.tsv',sep='\t')

data['time_stamp'] = pd.to_datetime(data['time_stamp'])

start = min(data['time_stamp'])
end = max(data['time_stamp'])

interval = (end - start) /2
data_train = data[data['time_stamp']<=start + interval]
data_val = data[data['time_stamp'] > start + interval]

start_train = min(data_train['time_stamp'])
end_train = max(data_train['time_stamp'])

interval_train = (end_train - start_train) /2

data_train_X = data_train[data_train['time_stamp'] <= start_train+interval_train]
data_train_y = data_train[data_train['time_stamp'] > start_train + interval_train]
