train_X, train_y = train_df[['mean','zc']], train_df['label']
test_X, test_y = test_df[['mean','zc']], test_df['label']

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(train_X)

train_X = sc.transform(train_X)
test_X = sc.transform(test_X)

model = OneClassSVM(nu=0.01)
model.fit(train_X)
pred = model.predict(test_X)
pred = np.where(pred == -1, 1, 0)
