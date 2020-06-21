from google.colab import drive
drive.mount('/content/drive')


from keras.layers import *
from keras.optimizers import Adam
from keras.models import Model, model_from_json
from keras.utils import plot_model
from keras.engine.topology import Network
import cv2
from keras.preprocessing import image as keras_image
from functions.instancenormalization import InstanceNormalization,  InputSpec
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

#===============================================================================
# functions for saving images

class TRAIN:
  def __init__(self):
    self.identity_mapping_modulus = 10  # Identity mapping will be done each time the iteration number is divisable with this number
    # Linear decay of learning rate, for both discriminators and generators
    self.use_linear_decay = False
    self.decay_epoch = 101  # The epoch where the linear decay of the learning rates start
    self.use_identity_learning = False
    self.batch_size = 1
    self.epochs = 200
    self.save_interval = 1

    self.train(self.epochs, batch_size=self.batch_size, save_interval = self.save_interval)
  def truncateAndSave(self, real_, real, synthetic, reconstructed, path_name):
    if len(real.shape) > 3:
        real = real[0]
        synthetic = synthetic[0]
        reconstructed = reconstructed[0]

    # Append and save
    if real_ is not None:
        if len(real_.shape) > 4:
            real_ = real_[0]
        image = np.hstack((real_[0], real, synthetic, reconstructed))
    else:
        image = np.hstack((real, synthetic, reconstructed))

    if channels == 1:
        image = image[:, :, 0]

    img = keras_image.array_to_img(image, scale=True)
    img.save(path_name)

  def saveImages(self, epoch, real_image_A, real_image_B, num_saved_images=1):
      directory = os.path.join('images', date_time)
      if not os.path.exists(os.path.join(directory, 'A')):
          os.makedirs(os.path.join(directory, 'A'))
          os.makedirs(os.path.join(directory, 'B'))
          os.makedirs(os.path.join(directory, 'Atest'))
          os.makedirs(os.path.join(directory, 'Btest'))

      testString = ''

      real_image_Ab = None
      real_image_Ba = None
      for i in range(num_saved_images + 1):
          if i == num_saved_images:
              real_image_A = A_test[0]
              real_image_B = B_test[0]
              real_image_A = np.expand_dims(real_image_A, axis=0)
              real_image_B = np.expand_dims(real_image_B, axis=0)
              testString = 'test'

          else:
              #real_image_A = A_train[rand_A_idx[i]]
              #real_image_B = B_train[rand_B_idx[i]]
              if len(real_image_A.shape) < 4:
                  real_image_A = np.expand_dims(real_image_A, axis=0)
                  real_image_B = np.expand_dims(real_image_B, axis=0)

          synthetic_image_B = G_A2B.predict(real_image_A)
          synthetic_image_A = G_B2A.predict(real_image_B)
          reconstructed_image_A = G_B2A.predict(synthetic_image_B)
          reconstructed_image_B = G_A2B.predict(synthetic_image_A)

          print('real_imagesA : real_imagesB')
          R=np.hstack((real_image_A[0], real_image_B[0]))
          plt.imshow(R),plt.show()

          print('synthetic_image_A : synthetic_image_B')
          syn = np.hstack((synthetic_image_A[0], synthetic_image_B[0]))
          plt.imshow(syn),plt.show()

          print('reconstructed_image_A : reconstructed_image_B')
          rec = np.hstack((reconstructed_image_A[0], reconstructed_image_B[0]))
          plt.imshow(rec),plt.show()

          #self.truncateAndSave(real_image_Ab, real_image_A, synthetic_image_B, reconstructed_image_A, 'images/{}/{}/epoch{}_sample{}.png'.format(self.date_time, 'A' + testString, epoch, i))
          #self.truncateAndSave(real_image_Ba, real_image_B, synthetic_image_A, reconstructed_image_B, 'images/{}/{}/epoch{}_sample{}.png'.format(self.date_time, 'B' + testString, epoch, i))


  #===============================================================================
  # Training
  def train(self, epochs, batch_size=1, save_interval=1):
      def run_training_iteration(loop_index, epoch_iterations):
          # ======= Discriminator training ==========
              # Generate batch of synthetic images
          synthetic_images_B = G_A2B.predict(real_images_A)
          synthetic_images_A = G_B2A.predict(real_images_B)
          synthetic_images_A = synthetic_pool_A.query(synthetic_images_A)
          synthetic_images_B = synthetic_pool_B.query(synthetic_images_B)

          for _ in range(discriminator_iterations):
              DA_loss_real = D_A.train_on_batch(x=real_images_A, y=ones)
              DB_loss_real = D_B.train_on_batch(x=real_images_B, y=ones)
              DA_loss_synthetic = D_A.train_on_batch(x=synthetic_images_A, y=zeros)
              DB_loss_synthetic = D_B.train_on_batch(x=synthetic_images_B, y=zeros)
              if use_multiscale_discriminator:
                  DA_loss = sum(DA_loss_real) + sum(DA_loss_synthetic)
                  DB_loss = sum(DB_loss_real) + sum(DB_loss_synthetic)
                  print('DA_losses: ', np.add(DA_loss_real, DA_loss_synthetic))
                  print('DB_losses: ', np.add(DB_loss_real, DB_loss_synthetic))
              else:
                  DA_loss = DA_loss_real + DA_loss_synthetic
                  DB_loss = DB_loss_real + DB_loss_synthetic
              D_loss = DA_loss + DB_loss

              if discriminator_iterations > 1:
                  print('D_loss:', D_loss)
                  sys.stdout.flush()

          # ======= Generator training ==========
          target_data = [real_images_A, real_images_B]  # Compare reconstructed images to real images
          if use_multiscale_discriminator:
              for i in range(2):
                  target_data.append(ones[i])
                  target_data.append(ones[i])
          else:
              target_data.append(ones)
              target_data.append(ones)

          if use_supervised_learning:
              target_data.append(real_images_A)
              target_data.append(real_images_B)

          for _ in range(generator_iterations):
              G_loss = G_model.train_on_batch(
                  x=[real_images_A, real_images_B], y=target_data)
              if generator_iterations > 1:
                  print('G_loss:', G_loss)
                  sys.stdout.flush()

          gA_d_loss_synthetic = G_loss[1]
          gB_d_loss_synthetic = G_loss[2]
          reconstruction_loss_A = G_loss[3]
          reconstruction_loss_B = G_loss[4]

          # Identity training
          if self.use_identity_learning and loop_index % self.identity_mapping_modulus == 0:
              G_A2B_identity_loss = G_A2B.train_on_batch(
                  x=real_images_B, y=real_images_B)
              G_B2A_identity_loss = G_B2A.train_on_batch(
                  x=real_images_A, y=real_images_A)
              print('G_A2B_identity_loss:', G_A2B_identity_loss)
              print('G_B2A_identity_loss:', G_B2A_identity_loss)

          # Update learning rates
          if self.use_linear_decay and epoch > self.decay_epoch:
              update_lr(D_A, decay_D)
              update_lr(D_B, decay_D)
              update_lr(G_model, decay_G)

          # Store training data
          DA_losses.append(DA_loss)
          DB_losses.append(DB_loss)
          gA_d_losses_synthetic.append(gA_d_loss_synthetic)
          gB_d_losses_synthetic.append(gB_d_loss_synthetic)
          gA_losses_reconstructed.append(reconstruction_loss_A)
          gB_losses_reconstructed.append(reconstruction_loss_B)

          GA_loss = gA_d_loss_synthetic + reconstruction_loss_A
          GB_loss = gB_d_loss_synthetic + reconstruction_loss_B
          D_losses.append(D_loss)
          GA_losses.append(GA_loss)
          GB_losses.append(GB_loss)
          G_losses.append(G_loss)
          reconstruction_loss = reconstruction_loss_A + reconstruction_loss_B
          reconstruction_losses.append(reconstruction_loss)

          print('Epoch----------------', epoch, '/', epochs)
          print('Loop index----------------', loop_index + 1, '/', epoch_iterations)
          print('D_loss: ', D_loss, 'G_loss: ', G_loss[0])
          print('reconstruction_loss: ', reconstruction_loss)
          print('DA_loss:', DA_loss, 'DB_loss:', DB_loss)

      # ======================================================================
      # Begin training
      # ======================================================================
      DA_losses = []
      DB_losses = []
      gA_d_losses_synthetic = []
      gB_d_losses_synthetic = []
      gA_losses_reconstructed = []
      gB_losses_reconstructed = []

      GA_losses = []
      GB_losses = []
      reconstruction_losses = []
      D_losses = []
      G_losses = []

      # Image pools used to update the discriminators
      synthetic_pool_A = ImagePool(synthetic_pool_size)
      synthetic_pool_B = ImagePool(synthetic_pool_size)

      # self.saveImages('(init)')

      # labels
      if use_multiscale_discriminator:
          label_shape1 = (batch_size,) + D_A.output_shape[0][1:]
          label_shape2 = (batch_size,) + D_A.output_shape[1][1:]
          #label_shape4 = (batch_size,) + self.D_A.output_shape[2][1:]
          ones1 = np.ones(shape=label_shape1) * REAL_LABEL
          ones2 = np.ones(shape=label_shape2) * REAL_LABEL
          #ones4 = np.ones(shape=label_shape4) * self.REAL_LABEL
          ones = [ones1, ones2]  # , ones4]
          zeros1 = ones1 * 0
          zeros2 = ones2 * 0
          #zeros4 = ones4 * 0
          zeros = [zeros1, zeros2]  # , zeros4]
      else:
          label_shape = (batch_size,) + D_A.output_shape[1:]
          ones = np.ones(shape=label_shape) * REAL_LABEL
          zeros = ones * 0

      # Linear decay
      if self.use_linear_decay:
          decay_D, decay_G = get_lr_linear_decay_rate()

      # Start stopwatch for ETAs
      start_time = time.time()

      for epoch in range(1, epochs + 1):
        random_order_A = np.random.randint(len(A_train), size=len(A_train))
        random_order_B = np.random.randint(len(B_train), size=len(B_train))
        epoch_iterations = max(len(random_order_A), len(random_order_B))
        min_nr_imgs = min(len(random_order_A), len(random_order_B))

        # If we want supervised learning the same images form
        # the two domains are needed during each training iteration
        if use_supervised_learning:
            random_order_B = random_order_A
        for loop_index in range(0, epoch_iterations, batch_size):
            if loop_index + batch_size >= min_nr_imgs:
                # If all images soon are used for one domain,
                # randomly pick from this domain
                if len(A_train) <= len(B_train):
                    indexes_A = np.random.randint(len(A_train), size=batch_size)
                    indexes_B = random_order_B[loop_index:
                                                loop_index + batch_size]
                else:
                  indexes_B = np.random.randint(len(B_train), size=batch_size)
                  indexes_A = random_order_A[loop_index:
                                                loop_index + batch_size]
            else:
                indexes_A = random_order_A[loop_index:
                                            loop_index + batch_size]
                indexes_B = random_order_B[loop_index:
                                            loop_index + batch_size]

            sys.stdout.flush()
            real_images_A = A_train[indexes_A]
            real_images_B = B_train[indexes_B]

            # Run all training steps
            run_training_iteration(loop_index, epoch_iterations)

        #================== within epoch loop end ==========================

        if epoch % save_interval == 0:
          print('\n', '\n', '-------------------------Saving images for epoch', epoch, '-------------------------', '\n', '\n')
          self.saveImages(epoch, real_images_A, real_images_B)

        if epoch % 20 == 0:
          G_model.save_weights('tb/G_model.h5')
          D_A.save_weights('tb/D_A.h5')
          D_B.save_weights('tb/D_B.h5')
          G_A2B.save_weights('tb/G_A2B.h5')
          G_B2A.save_weights('tb/G_B2A.h5')

        # Flush out prints each loop iteration
        sys.stdout.flush()

normalization = InstanceNormalization

# Hyper parameters
date_time_string_addition='_test'
img_shape = (256, 256, 3)
channels = img_shape[-1]
lambda_1 = 10.0  # Cyclic loss weight A_2_B
lambda_2 = 10.0  # Cyclic loss weight B_2_A
lambda_D = 1.0  # Weight for loss from discriminator guess on synthetic images
generator_iterations = 1  # Number of generator training iterations in each training loop
discriminator_iterations = 1  # Number of generator training iterations in each training loop
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
# Used as storage folder name
date_time = time.strftime('%Y%m%d-%H%M%S', time.localtime()) + date_time_string_addition

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
#D_A.load_weights("tb/D_A.h5")
#D_B.load_weights("tb/D_B.h5")

D_A.compile(optimizer=Adam(lr=2e-4, beta_1=0.5, beta_2 = 0.999),
                  loss=lse, loss_weights=loss_weights_D)
D_B.compile(optimizer=Adam(lr=2e-4, beta_1=0.5, beta_2 = 0.999),
                  loss=lse, loss_weights=loss_weights_D)
D_A.summary()

# Use Networks to avoid falsy keras error about weight descripancies
D_A_static = Network(inputs=image_A, outputs=guess_A, name='D_A_static_model')
D_B_static = Network(inputs=image_B, outputs=guess_B, name='D_B_static_model')

# ======= Generator model ==========
# Do note update discriminator weights during generator training
D_A_static.trainable = False
D_B_static.trainable = False

# Generators
G_A2B = GAN.modelGenerator(name='G_A2B_model')
G_B2A = GAN.modelGenerator(name='G_B2A_model')
#G_B2A.load_weights("tb/G_B2A.h5")
#G_A2B.load_weights("tb/G_A2B.h5")

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
#G_model.load_weights("tb/G_model.h5")
G_model.compile(optimizer=Adam(lr=2e-4, beta_1=0.5, beta_2 = 0.999),
                      loss=compile_losses, loss_weights=compile_weights)


G_B2A.summary()
G_model.summary()


print('--- loading data ---')
sys.stdout.flush()

image_folder='image'

#image_folder ="/Users/hagiharatatsuya/downloads/cycleGAN_img"

def trainGenerator(img_path):
    img=[]
    img_name=[]
    for idx, imgsp in enumerate(os.listdir(img_path)):
        imgs=cv2.imread(img_path+'/'+imgsp)
        if imgs is not None:
            imgs = cv2.resize(imgs, (256, 256))
            imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
            img.append(imgs/127.5-1)
            img_name.append(imgsp)
    inputs=np.array(img)
    return inputs, img_name

A_train, _ = trainGenerator(os.path.join(image_folder, 'trainA'))
B_train, _ = trainGenerator(os.path.join(image_folder, 'trainB'))
A_test, testA_image_names = trainGenerator(os.path.join(image_folder, 'testA'))
B_test, testB_image_names = trainGenerator(os.path.join(image_folder, 'testA'))


directory = os.path.join('images', date_time)
if not os.path.exists(directory):
    os.makedirs(directory)
    os.makedirs('tb')
train = TRAIN()
