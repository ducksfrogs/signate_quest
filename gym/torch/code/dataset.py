import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()

df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)

df_iris['target'] = iris.target
df_iris.loc[df_iris['target'] == 0, 'target'] = 'setosa'
df_iris.loc[df_iris['target'] == 1, 'target'] = 'versicolor'
df_iris.loc[df_iris['target'] == 2, 'target'] = 'virginica'
