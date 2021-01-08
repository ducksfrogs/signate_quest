
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D

model = Sequential()

model.add(Conv2D(filters=6, kernel_size=(3,3), input_shape=(96,96,3)))
model.add(Activation('sigmoid'))
model.add(MaxPooling2D(pool_size=(2,2))
model.add(Conv2D(filters=12, kernel_size=(3,3)))
