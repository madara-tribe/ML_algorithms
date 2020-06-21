from keras.layers import *
from functions.instancenormalization import InstanceNormalization,  InputSpec
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense
from keras.backend import mean
from keras.models import Model, model_from_json
from keras.engine.topology import Network
from functions.HelpFunction import *
from functions.ArchitectureFunction import *
import keras.backend as K
import tensorflow as tf



class cycleGAN(object):
  def __init__(self, img_shape):
    self.img_shape = img_shape
    # PatchGAN - if false the discriminator learning rate should be decreased
    self.use_patchgan = True
    self.use_multiscale_discriminator = False
    self.channels = img_shape[-1]

  def modelMultiScaleDiscriminator(self, name=None):
    x1 = Input(shape = self.img_shape)
    x2 = AveragePooling2D(pool_size=(2, 2))(x1)
    #x4 = AveragePooling2D(pool_size=(2, 2))(x2)

    out_x1 = self.modelDiscriminator('D1')(x1)
    out_x2 = self.modelDiscriminator('D2')(x2)
    #out_x4 = self.modelDiscriminator('D4')(x4)

    return Model(inputs=x1, outputs=[out_x1, out_x2], name=name)

  def modelDiscriminator(self, name=None):
      # Specify input
      input_img = Input(shape = self.img_shape)
      # Layer 1 (#Instance normalization is not used for this layer)
      x = ck(input_img, 64, False)
      # Layer 2
      x = ck(x, 128, True)
      # Layer 3
      x = ck(x, 256, True)
      # Layer 4
      x = ck(x, 512, True)
      # Output layer
      if self.use_patchgan:
          x = Conv2D(filters=1, kernel_size=4, strides=1, padding='same')(x)
      else:
          x = Flatten()(x)
          x = Dense(1)(x)
      x = Activation('sigmoid')(x)
      return Model(inputs=input_img, outputs=x, name=name)

  def modelGenerator(self, name=None):
      # Specify input
      input_img = Input(shape = self.img_shape)
      # Layer 1
      x = ReflectionPadding2D((3, 3))(input_img)
      x = c7Ak(x, 32)
      # Layer 2
      x = dk(x, 64)
      # Layer 3
      x = dk(x, 128)

      if self.use_multiscale_discriminator:
          # Layer 3.5
          x = dk(x, 256)

      # Layer 4-12: Residual layer
      for _ in range(4, 13):
          x = Rk(x)

      if self.use_multiscale_discriminator:
          # Layer 12.5
          x = uk(x, 128)

      # Layer 13
      x = uk(x, 64)
      # Layer 14
      x = uk(x, 32)
      x = ReflectionPadding2D((3, 3))(x)
      x = Conv2D(self.channels, kernel_size=7, strides=1)(x)
      x = Activation('tanh')(x)  # They say they use Relu but really they do not
      return Model(inputs=input_img, outputs=x, name=name)
