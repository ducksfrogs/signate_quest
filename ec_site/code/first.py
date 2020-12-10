import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv('input/train_B.tsv',sep='\t' )

ts = pd.to_datetime(data['time_stamp'])

Max = max(ts)
Min = min(ts)

data['event_'].value_counts()

data['time_stamp'] = ts
