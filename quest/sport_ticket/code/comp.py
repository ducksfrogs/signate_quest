import pandas as pd

train = pd.read_csv('../input/train.csv')
train_add =pd.read_csv('../input/train_add.csv')
condition = pd.read_csv('../input/condition.csv')
stadium = pd.read_csv('../input/stadium.csv')
add2014 = pd.read_csv('../input/2014_add.csv')

def get_month(x):
    return x[:2]

train['month'] = train['gameday'].apply(get_month)
data['month'] = data['gameday'].apply(get_month)

def get_week(x):
    return x[6:7]

data['week'] = data['gameday'].apply(get_week)

def get_match(x):
    return x[0:4]

def get_match(x):
    return x[x.find("第")+1:x.find("節")]

train['match_num'] = train['match'].apply(get_match)

data['match_num'] =data['match'].apply(get_match)

def get_humidity(x):
    return float(x[:-1])/100

data['humidity'] = data['humidity'].apply(get_humidity)

import matplotlib.pyplot as plt

matplotlib.pyplot.hist(data=y)
