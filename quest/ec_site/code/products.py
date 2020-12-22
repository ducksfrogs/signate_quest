
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv('input/train_B.tsv',sep='\t' )

ts = pd.to_datetime(data['time_stamp'])

data['time_stamp'] = ts


data_p = data.groupby('product_id').apply(lambda x: x.sort_values('time_stamp'))


data_p.loc['00000001_b']

users = data_p.loc['00000001_b']['user_id'].unique()

data_p.loc['00000001_b']['event_type'].value_counts()

data_p.loc['00000001_b']['time_stamp'].apply(lambda x: x.day_name()).value_counts())
