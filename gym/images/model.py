
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation

model = Sequential()
model.add(Dense(10, input_shape=(51, )))
model.add(Activation('tanh'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model = Sequential()
model.add(Dense(150,input_shape=(101,)))
model.add(Activation('sigmoid'))
model.add(Dense(200))
model.add(Activation('tanh'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()
