import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv('input/train_B.tsv',sep='\t' )

ts = pd.to_datetime(data['time_stamp'])

data['time_stamp'] = ts


data_up = data.groupby(['user_id','product_id']).apply(lambda x: x.sort_values('time_stamp'))

data_up.loc[('0000010_B','00010114_b')]
data_up.loc[('0000010_B','00010114_b')]['event_type'].value_counts()

data_up_0 = data_up.loc[('0000010_B','00010114_b')]

condition = data_up_0['event_type'] == 3

print(data_up_0[condition]['ad'].value_counts())

ymd = data_up.loc[('0000015_B', '00000744_b')]['time_stamp'].apply(lambda x: x.date()).unique()
