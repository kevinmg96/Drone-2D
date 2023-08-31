from keras import layers
from keras import initializers
from collections import namedtuple, deque
import random
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop 
from tensorflow import gather_nd
from keras.losses import mean_squared_error
import numpy as np
import numpy as np
from keras import Sequential

input = np.zeros([6,1])

model=Sequential()
model.add(Dense(10,input_dim = 6,activation='relu')) #añadir aqui state dimmension
model.add(Dense(10,activation='relu'))
model.add(Dense(4,activation='linear'))
print(model.summary())

#make prediction
num_test_samples = 5
test_data = np.random.random((num_test_samples, 6))
test_data = np.random.random(size=(1,6)) #column debe ser la dimension de la input layer model
print("testea la data : {}".format(test_data))
t = model.predict(test_data)

layer = layers.Dense(
    units=64,
    kernel_initializer=initializers.RandomNormal(stddev=0.01),
    bias_initializer=initializers.Zeros()
)

print(type(layer))