from keras.layers import *
from instancenormalization import InstanceNormalization,  InputSpec
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense
from keras.backend import mean
from keras.models import Model, model_from_json
from keras.utils import plot_model
from keras.engine.topology import Network
from ArchitectureFunction import *
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

normalization = InstanceNormalization

#===============================================================================
# Save and load

def load_model_and_generate_synthetic_images():
    response = input('Are you sure you want to generate synthetic images instead of training? (y/n): ')[0].lower()
    if response == 'y':
        load_model_and_weights(G_A2B)
        load_model_and_weights(G_B2A)
        synthetic_images_B = G_A2B.predict(A_test)
        synthetic_images_A = G_B2A.predict(B_test)

        def save_image(image, name, domain):
            if channels == 1:
                image = image[:, :, 0]
            toimage(image, cmin=-1, cmax=1).save(os.path.join(
                'generate_images', 'synthetic_images', domain, name))

        # Test A images
        for i in range(len(synthetic_images_A)):
            # Get the name from the image it was conditioned on
            name = testB_image_names[i].strip('.png') + '_synthetic.png'
            synt_A = synthetic_images_A[i]
            save_image(synt_A, name, 'A')

        # Test B images
        for i in range(len(synthetic_images_B)):
            # Get the name from the image it was conditioned on
            name = testA_image_names[i].strip('.png') + '_synthetic.png'
            synt_B = synthetic_images_B[i]
            save_image(synt_B, name, 'B')

        print('{} synthetic images have been generated and placed in ./generate_images/synthetic_images'
              .format(len(A_test) + len(B_test)))


#===============================================================================
# Test - simple model
def modelSimple(name=None):
    inputImg = Input(shape=img_shape)
    #x = Conv2D(1, kernel_size=5, strides=1, padding='same')(inputImg)
    #x = Dense(self.channels)(x)
    x = Conv2D(256, kernel_size=1, strides=1, padding='same')(inputImg)
    x = Activation('relu')(x)
    x = Conv2D(channels, kernel_size=1, strides=1, padding='same')(x)

    return Model(input=inputImg, output=x, name=name)


def trainSimpleModel():
    real_A = A_test[0]
    real_B = B_test[0]
    real_A = real_A[np.newaxis, :, :, :]
    real_B = real_B[np.newaxis, :, :, :]
    epochs = 200
    for epoch in range(epochs):
        print('Epoch {} started'.format(epoch))
        G_A2B.fit(x=A_train, y=B_train, epochs=1, batch_size=1)
        G_B2A.fit(x=B_train, y=A_train, epochs=1, batch_size=1)
        #loss = self.G_A2B.train_on_batch(x=real_A, y=real_B)
        #print('loss: ', loss)
        synthetic_image_A = G_B2A.predict(real_B, batch_size=1)
        synthetic_image_B = G_A2B.predict(real_A, batch_size=1)
        save_tmp_images(real_A, real_B, synthetic_image_A, synthetic_image_B)

    G_A2B.save_weights('tb/G_A2B-200.h5')
    G_B2A.save_weights('tb/G_B2A-200.h5')
