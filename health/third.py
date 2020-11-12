
df['AG_ratio'].fillna( df['Alb'] / (df['TP'] - df['Alb']), inplace=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

print(df.describe(include='all'))

col_categoric = ["Gender",'disease']
df_numeric = df.drop(col_categoric, axis=1)

df_categoric = df[col_categoric]

counts_disease = df_categoric["disease"].value_counts()

counts_disease.plot(kind='bar')

df_numeric.hist(figsize=(12,8))

plt.tight_layout()
