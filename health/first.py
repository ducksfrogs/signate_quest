import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('dataset.csv')

print(df.head(10))
print(df.shape)
print(df.info())

print(df.isnull().sum())

print(df[df.isnull().any(axis=1)]
