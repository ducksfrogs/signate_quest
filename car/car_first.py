import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('data.tsv', sep='\t')

data = data.drop(columns=['id'])

data = data.dropna()

plt.scatter(data['displacement'], data['kpl'])
plt.scatter(data['horsepower'], data['kpl'])
plt.scatter(data['acceleration'], data['kpl'])


sns.boxplot('drive_system', 'kpl', data=data)
