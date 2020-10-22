import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import *
from keras.optimizers import *
from models.pyramid_pooling_module import PyramidPoolingModule
from models.scse import channel_spatial_squeeze_excite

def conv_kernel_initializer(shape, dtype=K.floatx()):
    """Initialization for convolutional kernels.
    The main difference with tf.variance_scaling_initializer is that
    tf.variance_scaling_initializer uses a truncated normal with an uncorrected
    standard deviation, whereas here we use a normal distribution. Similarly,
    tf.contrib.layers.variance_scaling_initializer uses a truncated normal with
    a corrected standard deviation.
    Args:
        shape: shape of variable
        dtype: dtype of variable
    Returns:
        an initialization for the variable
    """
    kernel_height, kernel_width, _, out_filters = shape
    fan_out = int(kernel_height * kernel_width * out_filters)
    return tf.random_normal(
        shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


def create_model(inputs_):
  inputs = PyramidPoolingModule()(inputs_)
  o = Conv2D(32, (3, 3), padding='same', kernel_initializer=conv_kernel_initializer)(inputs)
  o = channel_spatial_squeeze_excite(o)
  o = MaxPooling2D((2, 2))(o)
  o = Conv2D(64, (3, 3), padding='same', kernel_initializer=conv_kernel_initializer)(o)
  o = channel_spatial_squeeze_excite(o)
  o = MaxPooling2D((2, 2))(o)
  o = Conv2D(64, (3, 3), padding='same', kernel_initializer=conv_kernel_initializer)(o)
  o = channel_spatial_squeeze_excite(o)
  #x = Dense(64, activation='relu')(x)
  o = Flatten()(o)
  o = Dense(2, activation='softmax')(o)
  model = Model(inputs=inputs_, outputs=o)
  model.compile(optimizer='adam', loss='categorical_crossentropy')
  return model
