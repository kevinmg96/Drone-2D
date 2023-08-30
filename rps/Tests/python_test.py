from tensorflow.keras import layers
from tensorflow.keras import initializers
import numpy as np
r = np.array([1,2,3])
if len(r.shape) > 1:
    print("hola")
print(r[:,0])

layer = layers.Dense(
    units=64,
    kernel_initializer=initializers.RandomNormal(stddev=0.01),
    bias_initializer=initializers.Zeros()
)

print(type(layer))