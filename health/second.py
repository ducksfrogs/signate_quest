import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('dataset.csv')

print(df.head(10))
print(df.shape)
print(df.info())

print(df.isnull().sum())

print(df[df.isnull().any(axis=1)]

df['AG_ratio'].fillna( df['Alb'] / (df['TP'] - df['Alb']), inplace=True)

print(df.loc[[207,239,251,310],:])

print(df.duplicated().sum())
print(df[df.duplicated()])

df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
