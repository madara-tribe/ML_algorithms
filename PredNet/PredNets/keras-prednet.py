import os
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.callbacks import *
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam, SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras import activations
from keras.layers import *
from keras.engine import InputSpec
from tensorflow.python.client import device_lib
import cv2
import warnings
import matplotlib
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
device_lib.list_local_devices()



use_im_num = 300
im_size = [64,64]


# A unit
x_input = Input(shape=(im_size[0],im_size[1],1),dtype='float32', name='x_input')
A = Conv2D(1, (3, 3), padding='same',name='A')(x_input)

# R unit 
e_input = Input(shape=(1,im_size[0],im_size[1], 1),dtype='float32', name='e_input')
r_input = Input(shape=(im_size[0],im_size[1], 1),dtype='float32', name='r_input')
R = ConvLSTM2D(filters=1, kernel_size=(4, 4),
                   input_shape=(1, im_size[0],im_size[1], 1),
                   padding='same', return_sequences=False)(e_input)
Ahat = Conv2D(1, (3, 3), padding='same',name='Ahat')(R)

# E unit : pixcel layer (l=0) 
Relu = Activation('relu',name='Relu')
e0 = Relu(Lambda(lambda x: x[0] - x[1],output_shape=(im_size[0],im_size[1], 1),name='X-Ahat')([x_input,Ahat]))
e1 = Relu(Lambda(lambda x: x[1] - x[0],output_shape=(im_size[0],im_size[1], 1),name='Ahat-X')([x_input,Ahat]))
E = add([e0,e1])

# Model 
predict_m = Model(input=[e_input],output=[Ahat])
predict_m.compile(optimizer='rmsprop', loss='mean_squared_error')
predict_m.summary()


model = Model(input=[x_input,e_input],output=[E])
model.compile(optimizer='rmsprop', loss='mean_squared_error')
model.summary()


# Data files
train_file = 'train_image.npy'

# Training parameters
batch_size = 1
nt = 120  # number of timesteps used for sequences in training



#X=np.load(train_file)  # (17487, 420, 340)
plt.imshow(X[1], "gray"),plt.show()
X = [cv2.resize(img, (64, 64)) for img in X[:1000]]
X = np.array(X).reshape(len(X), 64, 64, 1)
plt.imshow(X[1].reshape(64, 64), "gray"),plt.show()
print(X.shape)
print(X.min(), X.max())



def get_output_layer(model, layer_name,n):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name].get_output_at(n)
    return layer          

          
in_E = np.zeros((1,im_size[0],im_size[1], 1))
labels = np.zeros((1,im_size[0],im_size[1], 1))
print(in_E.shape, labels.shape)


for epoch in range(1000):    
    image_num = 0
    batch_idx= min(len(X), np.inf) // 100
    for im_num in range(len(X)):
        print(epoch,im_num)
        in_im = X[im_num].reshape((1,im_size[0],im_size[1], 1))
        
        in_E = in_E.reshape((1,1,im_size[0],im_size[1], 1))
        model.fit([in_im,in_E], labels)
        in_E = model.predict([in_im, in_E])
        tmp_E = in_E.reshape((1,1,im_size[0],im_size[1], 1))
        out_im = predict_m.predict(tmp_E)
        print(in_E.shape)
        image_num += 1
        
        if image_num > use_im_num:
            break

        print("show result")
        predict_image = get_output_layer(model, 'Ahat',0)
        print(out_im.shape)
        show_im = (out_im[0].reshape(im_size[0],im_size[1]))
        plt.imshow(show_im),plt.show()
