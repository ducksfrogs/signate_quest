import pandas as pd

data = pd.read_csv('input/train_B.tsv', sep='\t')

u_p = data_train_X.groupby('user_id').apply(lambda x: len(x['project_id'].unique()))

u_d = data_train_X.groupby('user_id').apply(lambda x: len(le)
