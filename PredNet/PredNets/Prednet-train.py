import os
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.callbacks import *
from keras.layers import TimeDistributed
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam, SGD
from Residual_Prednet import PredNet
from utils.DataGenerator import SequenceGenerator
from tensorflow.python.client import device_lib
import cv2
import warnings
import matplotlib
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
device_lib.list_local_devices()



# Data files
train_file = 'train_images.npy'

# Training parameters

H=160
W=128
batch_size = 1
nt = 120  # number of timesteps used for sequences in training


# In[5]:


X=np.load(train_file)  # (17487, 420, 340)
plt.imshow(X[1], "gray"),plt.show()
X = [cv2.resize(img, (W, H)) for img in X]
X = np.array(X).reshape(len(X), H, W, 1)
plt.imshow(X[1].reshape(H, W), "gray"),plt.show()
print(X.shape)
print(X.min(), X.max())


train_img=X[:17000-(nt+1)]
valid_img = X[17000:-(nt+1)]
print(train_img.shape, valid_img.shape)


# prednet 


n_channels, im_height, im_width = (1, H, W)
input_shape = (im_height, im_width, n_channels)
stack_sizes = (n_channels, 48, 96, 192)
R_stack_sizes = stack_sizes
A_filt_sizes = (3, 3, 3)
Ahat_filt_sizes = (3, 3, 3, 3)
R_filt_sizes = (3, 3, 3, 3)
layer_loss_weights = np.array([1., 0., 0., 0.])  # weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
time_loss_weights = 1./ (nt - 1) * np.ones((nt,1))  # equally weight all timesteps except the first
time_loss_weights[0] = 0

prednet = PredNet(stack_sizes, R_stack_sizes,
                  A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                  output_mode='error', return_sequences=True)



inputs = Input(shape=(nt,) + input_shape)
errors = prednet(inputs)  # errors will be (batch_size, nt, nb_layers)
errors_by_time = TimeDistributed(Dense(1, trainable=False), weights=[layer_loss_weights, np.zeros(1)], trainable=False)(errors)  # calculate weighted error by layer
errors_by_time = Flatten()(errors_by_time)  # will be (batch_size, nt)
final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)  # weight errors by time
model = Model(inputs=inputs, outputs=final_errors)
model.compile(loss='mean_absolute_error', optimizer=Adam(lr=0.001))
# model.load_weights("09_prednet.hdf5")
model.summary()


# callback


lr_schedule = lambda epoch: 0.001 if epoch < 75 else 0.0001    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
callback = [LearningRateScheduler(lr_schedule)]
callback.append(ModelCheckpoint('{epoch:02d}_prednet.hdf5', monitor='loss', verbose=1))

train_generator = SequenceGenerator(train_img, H, W, nt, batch_size)
train_gene =train_generator.flow_from_img()
val_generator = SequenceGenerator(valid_img, H, W, nt, batch_size)
val_gene =val_generator.flow_from_img()


# train
history = model.fit_generator(train_gene, steps_per_epoch=1500, epochs=10, callbacks=callback,
                validation_data=val_gene, validation_steps=300)
