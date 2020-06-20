import os
import numpy as np
import tensorflow as tf
import pandas as pd
from natsort import natsorted
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.callbacks import *
from keras.layers import TimeDistributed
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam
from Residual_Prednet import PredNet
from tensorflow.python.client import device_lib
import cv2
import warnings
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings('ignore')
device_lib.list_local_devices()


def extrap_loss(y_true, y_hat):
    y_true = y_true[:, 1:]
    y_hat = y_hat[:, 1:]
    return 0.5 * K.mean(K.abs(y_true - y_hat), axis=-1)  # 0.5 to match scale of loss when trained in error mode (positive and negative errors split)

class TrainParams(Enum):
    ORIG_WEIGHT="07_prednet.hdf5"
    EXTRAP_WEIGHT = "44_ExtrapFinetune.hdf5"


    
H=160
W=128
extrap_start_time = 96  # starting at this time step, the prediction from the previous time step will be treated as the actual input
batch_size = 1
nt = 120

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


# In[4]:


inputs = Input(shape=(nt,) + input_shape)
errors = prednet(inputs)  # errors will be (batch_size, nt, nb_layers)
errors_by_time = TimeDistributed(Dense(1, trainable=False), weights=[layer_loss_weights, np.zeros(1)], trainable=False)(errors)  # calculate weighted error by layer
errors_by_time = Flatten()(errors_by_time)  # will be (batch_size, nt)
final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)  # weight errors by time
orig_model = Model(inputs=inputs, outputs=final_errors)
orig_model.compile(loss='mean_absolute_error', optimizer='adam')
orig_model.load_weights(TrainParams.ORIG_WEIGHT)
orig_model.summary()


# In[5]:


layer_config = orig_model.layers[1].get_config()
layer_config['output_mode'] = 'prediction'
layer_config['extrap_start_time'] = extrap_start_time
data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
prednet = PredNet(weights=orig_model.layers[1].get_weights(), **layer_config)

input_shape = list(orig_model.layers[0].batch_input_shape[1:])
input_shape[0] = nt
inputs = Input(input_shape)
predictions = prednet(inputs)
extrap_model = Model(inputs=inputs, outputs=predictions)
extrap_model.compile(loss=extrap_loss, optimizer=Adam(lr=0.0001))
extrap_model.load_weights(TrainParams.EXTRAP_WEIGHT)
extrap_model.summary()



span_dir= "inference_terms"
paths= [path for path in natsorted(os.listdir(span_dir))]

indexes = 1
span1=paths[indexes]
sp = np.load(os.path.join(span_dir,span1))
sp = np.array([cv2.resize(img, (W, H)) for img in sp])
print(span1, sp.shape, np.unique(sp[1]))

pred_span=nt-extrap_start_time
X_all = sp.reshape(1, extrap_start_time, H, W, 1)
zeros = np.zeros((1, nt-extrap_start_time) + (H, W, 1), np.float32)
print(X_all.shape, zeros.shape)
TX=np.hstack([X_all, zeros])
TX=TX/255
print(TX.shape, TX.min(), TX.max())

# predict
TX_hat = extrap_model.predict(TX, batch_size)
print(TX.shape, TX_hat.shape)



TX_hats=[]
for hat in TX_hat[0][96:]:
    TX_hats.append(hat)
print(len(TX_hats))


# concat
h1=cv2.resize(TX_hats[5].reshape(H, W), (340, 420))
h2=cv2.resize(TX_hats[11].reshape(H, W), (340, 420))
h3=cv2.resize(TX_hats[17].reshape(H, W), (340, 420))
h4=cv2.resize(TX_hats[23].reshape(H, W), (340, 420))
plt.imshow(h1, 'gray'),plt.show()
plt.imshow(h2, 'gray'),plt.show()
plt.imshow(h3, 'gray'),plt.show()
plt.imshow(h4, 'gray'),plt.show()
print(h1.shape, h2.shape, h3.shape, h4.shape)

# save
pred_span1=np.vstack([h1, h2, h3, h4])
plt.imshow(pred_span1, 'gray'),plt.show()
print(pred_span1.shape)
np.save('submit_csv/span2', pred_span1)



""" cancat all span data """
from natsort import natsorted

dir_path='submit_csv'
load_path = [os.path.join(dir_path, path) for path in os.listdir(dir_path)]
sorted_path =[path for path in natsorted(load_path)]
print(sorted_path, len(sorted_path))

Xs=[]
for idx, npy in enumerate(sorted_path):
    img = np.load(npy)
    img = (img*255).astype('int32')
    img[img<0]=0
    print(idx, npy, img.shape, img.min(), img.max())
    if idx<2:
        plt.imshow(img, 'gray'),plt.show()
    Xs.append(img)


submit_csvs=np.vstack(Xs[0:])
print(submit_csvs.shape, submit_csvs.min(), submit_csvs.max())

# make csv for submint
df=pd.DataFrame(submit_csvs)
df.to_csv('submit.csv', header=False, index=True)
