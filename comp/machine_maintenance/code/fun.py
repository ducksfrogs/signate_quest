import keras.models
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Activation
from keras.models import Sequential

inputDim = 40064
model = Sequential()

model.add(Dense(64, input_shape=(inputDim,)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(8))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(inputDim))

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(x=trainX, y=trainX, epochs=10,batch_size=50, validation_split=0.1)

trainX = train_df
testX, testY = test_df.drop(columns=['label']), test_df['label']

model = keras.models.load_model('model.h5', compile=False)
train_pred = model.predict(trainX)

train_sscore = np.mean(np.square(trainX - train_pred))

train_score.plot.hist()

test_pred = model.predict(testX)

test_score = np.mean(np.square(testX - test_pred), axis=1)

threshold = train_score.quantile(0.8)

pred = np.where(test_score > threshold, 1, 0)
