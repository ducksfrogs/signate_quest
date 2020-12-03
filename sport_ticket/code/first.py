import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

sample = pd.read_csv('sample.csv')

print(sample)

age = sample[sample['名前']=='yamada']['年齢']

data = pd.read_csv('data.csv')
