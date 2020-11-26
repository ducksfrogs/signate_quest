import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from  tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV



df = pd.read_csv('../data/train.csv', index_col=0, na_values='?')
test = pd.read_csv('../data/test.csv', index_col=0, na_values='?')

mean = np.mean(df['horsepower'])

df['horsepower'].fillna(mean, inplace=True)


train = df[['cylinders',
             'displacement',
             'horsepower',
             'weight',
             'acceleration',
             'model year',
             'origin']]

Y = df[['mpg']]

trainX, testX, trainY, testY = train_test_split(train, Y)

mean = np.mean(trainX['horsepower'])
trainX['horsepower'].fillna(mean, inplace=True)

testX['horsepower'].fillna(mean, inplace=True)



origin = df.pop('origin')
df['USA'] = (origin==1) * 1.0
df['Europe']= (origin==2) * 1.0
df['Japan']= (origin==3) * 1.0

scores = []
reg = RandomForestRegressor()

RFR_grid = {reg:{"n_estimators": [i for i in range(1,21)],
                 "criterion": ["gini", "entropy"],
                 "max_depth":[i for i in range(1,5)],
                 "random_state": [i for i in range(0,10)]}}

for model, param in tqdm(RFR_grid.items()):
    reg = GridSearchCV(model, param)
    reg.fit(trainX, trainY)
    score = reg.score(testX, testY)
    scores.append(score)
    
