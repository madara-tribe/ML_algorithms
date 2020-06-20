import os
import numpy as np
import tensorflow as tf
from keras.utils import multi_gpu_model
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.callbacks import *
from keras.layers import TimeDistributed
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam
from Residual_Prednet import PredNet
from utils.DataGenerator import SequenceGenerator
from tensorflow.python.client import device_lib
import cv2
import warnings
import matplotlib
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
device_lib.list_local_devices()


def extrap_loss(y_true, y_hat):
    y_true = y_true[:, 1:]
    y_hat = y_hat[:, 1:]
    return 0.5 * K.mean(K.abs(y_true - y_hat), axis=-1)  # 0.5 to match scale of loss when trained in error mode (positive and negative errors split)

train_file = 'train_image.npy'
weights_file="first_weight/07_prednet.hdf5"


# Training parameters
H=160
W=128
extrap_start_time = 96  # starting at this time step, the prediction from the previous time step will be treated as the actual input
batch_size = 1
nt = 120


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



inputs1 = Input(shape=(nt,) + input_shape)
errors = prednet(inputs1)  # errors will be (batch_size, nt, nb_layers)
errors_by_time = TimeDistributed(Dense(1, trainable=False), weights=[layer_loss_weights, np.zeros(1)], trainable=False)(errors)  # calculate weighted error by layer
errors_by_time = Flatten()(errors_by_time)  # will be (batch_size, nt)
final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)  # weight errors by time
orig_model = Model(inputs=inputs1, outputs=final_errors)
orig_model.compile(loss='mean_absolute_error', optimizer=Adam(lr=0.001))
orig_model.load_weights(weights_file)
orig_model.summary()


layer_config = orig_model.layers[1].get_config()
layer_config['output_mode'] = 'prediction'
layer_config['extrap_start_time'] = extrap_start_time
data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
prednet = PredNet(weights=orig_model.layers[1].get_weights(), **layer_config)

input_shape = list(orig_model.layers[0].batch_input_shape[1:])
input_shape[0] = nt
inputs = Input(input_shape)
predictions = prednet(inputs)
model = Model(inputs=inputs, outputs=predictions)
# SGD(lr=0.01, momentum=0.9, decay=0.001, nesterov=True)
model.compile(loss=extrap_loss, optimizer=Adam(lr=0.0001))
model.load_weights("05_ExtrapFinetune.hdf5")
model.summary()


# generator


lr_schedule = lambda epoch: 0.001 if epoch < 4 else 0.0001    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
callback = [LearningRateScheduler(lr_schedule)]
callback.append(ModelCheckpoint('{epoch:02d}_ExtrapFinetune.hdf5', monitor='loss', verbose=1))

train_generator = SequenceGenerator(train_img, H, W, nt, batch_size, output_mode='prediction')
train_gene =train_generator.flow_from_img()
val_generator = SequenceGenerator(valid_img, H, W, nt, batch_size, output_mode='prediction')
val_gene =val_generator.flow_from_img()


# train


try:
    model.fit_generator(train_gene, steps_per_epoch=140, epochs=20, callbacks=callback,
                validation_data=val_gene, validation_steps=20)
finally:
    model.save('65_ExtrapFinetune.hdf5')
