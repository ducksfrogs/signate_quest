import pandas as pd

data = pd.read_csv('input/train_B.tsv', sep='\t')

u_p = data_train_X.groupby('user_id').apply(lambda x: len(x['product_id'].unique()))
u_d = data_train_X.groupby('user_id').apply(lambda x: len(x['time_stamp'].apply(lambda x: x.date()).unique()))
u_pv = data_train_X.groupby('user_id').apply(lambda x: (x['event_type']==1).sum())

u =pd.concat([u_p, u_d, u_pv], axis=1)

u.columns = ['u_p', 'u_d', 'u_pv']


p_u = data_train_X.groupby('product_id').apply(lambda x: len(x['user_id'].unique()))
p_ca = data_train_X.groupby('product_id').apply(lambda x: (x['event_type']==0).sum())
p_pv = data_train_X.groupby('product_id').apply(lambda x: )
