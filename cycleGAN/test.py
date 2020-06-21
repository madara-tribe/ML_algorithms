from google.colab import drive
drive.mount('/content/drive')

from keras.optimizers import Adam
import cv2
from functions.ArchitectureFunction import *
from functions.HelpFunction import *
from cycleGAN.cycleGAN import cycleGAN
import numpy as np
import random
import datetime
import time
import math
import csv
import sys
import os
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf


def trainGenerator(img_path):
    img=[]
    img_name=[]
    for idx, imgsp in enumerate(os.listdir(img_path)):
        imgs=cv2.imread(img_path+'/'+imgsp)
        if imgs is not None:
            imgs = cv2.resize(imgs, (256, 256))
            imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
            imgs = imgs.reshape(1, 256, 256, 3)
            img.append(imgs/127.5-1)
            img_name.append(imgsp)
    inputs = np.array(img)
    return inputs, img_name

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:4]
    image = np.zeros((height*shape[0], width*shape[1], shape[2]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1],:] = img[:, :, :]
    return image

def predict():
  # Hyper parameters
  img_shape = (256, 256, 3)
  channels = img_shape[-1]
  lambda_1 = 10.0  # Cyclic loss weight A_2_B
  lambda_2 = 10.0  # Cyclic loss weight B_2_A
  lambda_D = 1.0  # Weight for loss from discriminator guess on synthetic images

  synthetic_pool_size = 50
  # Multi scale discriminator - if True the generator have an extra encoding/decoding step to match discriminator information access
  use_multiscale_discriminator = False
  # Resize convolution - instead of transpose convolution in deconvolution layers (uk) - can reduce checkerboard artifacts but the blurring might affect the cycle-consistency
  use_resize_convolution = False
  # Supervised learning part - for MR images - comparison
  use_supervised_learning = False
  use_identity_learning = False
  supervised_weight = 10.0
  REAL_LABEL = 1.0  # Use e.g. 0.9 to avoid training the discriminators to zero loss


  GAN = cycleGAN(img_shape)
  # ======= Discriminator model ==========
  if use_multiscale_discriminator:
      D_A = GAN.modelMultiScaleDiscriminator()
      D_B = GAN.modelMultiScaleDiscriminator()
      loss_weights_D = [0.5, 0.5] # 0.5 since we train on real and synthetic images
  else:
      D_A = GAN.modelDiscriminator()
      D_B = GAN.modelDiscriminator()
      loss_weights_D = [0.5]  # 0.5 since we train on real and synthetic images


  # Discriminator builds
  image_A = Input(shape=img_shape)
  image_B = Input(shape=img_shape)
  guess_A = D_A(image_A)
  guess_B = D_B(image_B)
  D_A = Model(inputs=image_A, outputs=guess_A, name='D_A_model')
  D_B = Model(inputs=image_B, outputs=guess_B, name='D_B_model')
  D_A.load_weights("drive/My Drive/GANtb/D_A.h5")
  D_B.load_weights("drive/My Drive/GANtb/D_B.h5")

  D_A.compile(optimizer=Adam(lr=2e-4, beta_1=0.5, beta_2 = 0.999),
                    loss=lse, loss_weights=loss_weights_D)
  D_B.compile(optimizer=Adam(lr=2e-4, beta_1=0.5, beta_2 = 0.999),
                    loss=lse, loss_weights=loss_weights_D)
  D_A.summary()

  # Use Networks to avoid falsy keras error about weight descripancies
  D_A_static = Network(inputs=image_A, outputs=guess_A, name='D_A_static_model')
  D_B_static = Network(inputs=image_B, outputs=guess_B, name='D_B_static_model')

  # ======= Generator model ==========

  # Generators
  G_A2B = GAN.modelGenerator(name='G_A2B_model')
  G_B2A = GAN.modelGenerator(name='G_B2A_model')
  G_B2A.load_weights("drive/My Drive/GANtb/G_B2A.h5")
  G_A2B.load_weights("drive/My Drive/GANtb/G_A2B.h5")

  # self.G_A2B.summary()
  if use_identity_learning:
      G_A2B.compile(optimizer=Adam(lr=2e-4, beta_1=0.5, beta_2 = 0.999), loss='MAE')
      G_B2A.compile(optimizer=Adam(lr=2e-4, beta_1=0.5, beta_2 = 0.999), loss='MAE')

  # Generator builds
  real_A = Input(shape=img_shape, name='real_A')
  real_B = Input(shape=img_shape, name='real_B')
  synthetic_B = G_A2B(real_A)
  synthetic_A = G_B2A(real_B)
  dA_guess_synthetic = D_A_static(synthetic_A)
  dB_guess_synthetic = D_B_static(synthetic_B)
  reconstructed_A = G_B2A(synthetic_B)
  reconstructed_B = G_A2B(synthetic_A)

  model_outputs = [reconstructed_A, reconstructed_B]
  compile_losses = [cycle_loss, cycle_loss, lse, lse]
  compile_weights = [lambda_1, lambda_2, lambda_D, lambda_D]

  if use_multiscale_discriminator:
      for _ in range(2):
          compile_losses.append(lse)
          compile_weights.append(lambda_D)  # * 1e-3)  # Lower weight to regularize the model
      for i in range(2):
          model_outputs.append(dA_guess_synthetic[i])
          model_outputs.append(dB_guess_synthetic[i])
  else:
      model_outputs.append(dA_guess_synthetic)
      model_outputs.append(dB_guess_synthetic)

  if use_supervised_learning:
      model_outputs.append(synthetic_A)
      model_outputs.append(synthetic_B)
      compile_losses.append('MAE')
      compile_losses.append('MAE')
      compile_weights.append(supervised_weight)
      compile_weights.append(supervised_weight)

  G_model = Model(inputs=[real_A, real_B], outputs=model_outputs, name='G_model')
  G_model.load_weights("drive/My Drive/GANtb/G_model.h5")
  G_model.compile(optimizer=Adam(lr=2e-4, beta_1=0.5, beta_2 = 0.999),
                        loss=compile_losses, loss_weights=compile_weights)
  
  ### Predict image and polot combined images ###
  image_folder='/content/drive/My Drive'
  A_train, _ = trainGenerator(os.path.join(image_folder, 'trainA'))

  domainB = np.array([G_A2B.predict(img) for img in A_train])
  domainBs = domainB.reshape(50, 256, 256, 3)
  print(domainBs.shape)
  combined_img = combine_images(domainBs)
  plt.imshow(combined_img),plt.show()

if __name__ == '__main__':
  predict()
