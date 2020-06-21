from keras.layers import *
from instancenormalization import InstanceNormalization,  InputSpec
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense
from keras.backend import mean
from keras.models import Model, model_from_json
from keras.utils import plot_model
from keras.engine.topology import Network
from HelpFunction import *
from keras.preprocessing.image import save_img
import numpy as np
import random
import datetime
import time
import math
import sys
import os
import keras.backend as K
import tensorflow as tf


# Identity loss - sometimes send images from B to G_A2B (and the opposite) to teach identity mappings
use_identity_learning = False
use_resize_convolution = False
normalization = InstanceNormalization

#===============================================================================
# Architecture functions

def ck(x, k, use_normalization):
    x = Conv2D(filters=k, kernel_size=4, strides=2, padding='same')(x)
    # Normalization is not done on the first discriminator layer
    if use_normalization:
        x = normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
    x = LeakyReLU(alpha=0.2)(x)
    return x

def c7Ak(x, k):
    x = Conv2D(filters=k, kernel_size=7, strides=1, padding='valid')(x)
    x = normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
    x = Activation('relu')(x)
    return x

def dk(x, k):
    x = Conv2D(filters=k, kernel_size=3, strides=2, padding='same')(x)
    x = normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
    x = Activation('relu')(x)
    return x

def Rk(x0):
    k = int(x0.shape[-1])
    # first layer
    x = ReflectionPadding2D((1,1))(x0)
    x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid')(x)
    x = normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
    x = Activation('relu')(x)
    # second layer
    x = ReflectionPadding2D((1, 1))(x)
    x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid')(x)
    x = normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
    # merge
    x = add([x, x0])
    return x

def uk(x, k):
    # (up sampling followed by 1x1 convolution <=> fractional-strided 1/2)
    if use_resize_convolution:
        x = UpSampling2D(size=(2, 2))(x)  # Nearest neighbor upsampling
        x = ReflectionPadding2D((1, 1))(x)
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid')(x)
    else:
        x = Conv2DTranspose(filters=k, kernel_size=3, strides=2, padding='same')(x)  # this matches fractinoally stided with stride 1/2
    x = normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
    x = Activation('relu')(x)
    return x


def lse(y_true, y_pred):
    loss = tf.reduce_mean(tf.squared_difference(y_pred, y_true))
    return loss

def cycle_loss(y_true, y_pred):
    loss = tf.reduce_mean(tf.abs(y_pred - y_true))
    return loss

def get_lr_linear_decay_rate():
    # Calculate decay rates
    max_nr_images = max(len(A_train), len(B_train))

    updates_per_epoch_D = 2 * max_nr_images + discriminator_iterations - 1
    updates_per_epoch_G = max_nr_images + generator_iterations - 1
    if use_identity_learning:
        updates_per_epoch_G *= (1 + 1 / identity_mapping_modulus)
    denominator_D = (epochs - decay_epoch) * updates_per_epoch_D
    denominator_G = (epochs - decay_epoch) * updates_per_epoch_G
    decay_D = learning_rate_D / denominator_D
    decay_G = learning_rate_G / denominator_G

    return decay_D, decay_G

def update_lr(model, decay):
    new_lr = K.get_value(model.optimizer.lr) - decay
    if new_lr < 0:
        new_lr = 0
    # print(K.get_value(model.optimizer.lr))
    K.set_value(model.optimizer.lr, new_lr)



# reflection padding taken from
# https://github.com/fastai/courses/blob/master/deeplearning2/neural-style.ipynb
class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            if len(image.shape) == 3:
                image = image[np.newaxis, :, :, :]

            if self.num_imgs < self.pool_size:  # fill up the image pool
                self.num_imgs = self.num_imgs + 1
                if len(self.images) == 0:
                    self.images = image
                else:
                    self.images = np.vstack((self.images, image))

                if len(return_images) == 0:
                    return_images = image
                else:
                    return_images = np.vstack((return_images, image))

            else:  # 50% chance that we replace an old synthetic image
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id, :, :, :]
                    tmp = tmp[np.newaxis, :, :, :]
                    self.images[random_id, :, :, :] = image[0, :, :, :]
                    if len(return_images) == 0:
                        return_images = tmp
                    else:
                        return_images = np.vstack((return_images, tmp))
                else:
                    if len(return_images) == 0:
                        return_images = image
                    else:
                        return_images = np.vstack((return_images, image))

        return return_images
