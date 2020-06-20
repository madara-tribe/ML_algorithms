from google.colab import drive
drive.mount('/content/drive')

import os
import numpy as np
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, TimeDistributed, Flatten
from keras.layers.recurrent import LSTM
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.callbacks import *
from keras.layers.wrappers import Bidirectional
from keras.layers import Input
import math
import cv2
import tensorflow as tf

train=np.load('drive/My Drive/train.npy')
valid=np.load('drive/My Drive/valid.npy')
train =np.array([cv2.resize(imgs, (85, 105)) for imgs in train])
valid = np.array([cv2.resize(imgs, (85, 105)) for imgs in valid])
train = train/127.5-1
valid = valid/127.5-1
print(train.shape, valid.shape)  # (4147, 105, 85) (612, 105, 85)  /4
print(train.min(), train.max())
plt.imshow(train[1], 'gray'),plt.show()
plt.imshow(valid[1], 'gray'),plt.show()

Xs = train.flatten().reshape(len(train), 105*85)
ys = valid.flatten().reshape(len(valid), 105*85)
print(Xs.shape, ys.shape)  # (4147, 8925) (612, 8925)

# 次元数
DIMENSION = 8925
LENGTH_PER_UNIT = 48  # 長さはいくらでもいい、予測期間は長さ以下にできるので24にして24h予測する

# for train
# 時系列データを入れる箱
sequences = []
# 正解データを入れる箱
target = []
# for i in range(0, Xs.shape[0]-LENGTH_PER_UNIT):
for i in range(0, int(Xs.shape[0]/LENGTH_PER_UNIT)):
  print('X', i, i+LENGTH_PER_UNIT)
  sequences.append(Xs[i:i + LENGTH_PER_UNIT])
  print('Y', i+LENGTH_PER_UNIT)
  target.append(Xs[i+LENGTH_PER_UNIT])

# データを成形
X = np.array(sequences).reshape(len(sequences), LENGTH_PER_UNIT, DIMENSION)
Y = np.array(target).reshape(len(target), DIMENSION)
print(X.shape, Y.shape)  # (86, 48, 8925) (86, 8925)

# for validation
# 時系列データを入れる箱
valid_seq = []
# 正解データを入れる箱
valid_target = []
# for i in range(0, Xs.shape[0]-LENGTH_PER_UNIT):
for i in range(0, int(ys.shape[0]/LENGTH_PER_UNIT)):
  print('X', i, i+LENGTH_PER_UNIT)
  valid_seq.append(ys[i:i + LENGTH_PER_UNIT])
  print('Y', i+LENGTH_PER_UNIT)
  valid_target.append(ys[i+LENGTH_PER_UNIT])

# データを成形
Xv = np.array(valid_seq).reshape(len(valid_seq), LENGTH_PER_UNIT, DIMENSION)
yv = np.array(valid_target).reshape(len(valid_target), DIMENSION)
print(Xv.shape, yv.shape)  # (12, 48, 8925) (12, 8925)

class SequenceGenerator(object):
    def __init__(self, x_img, y_img):
        self.X = x_img
        self.y = y_img
        self.batch_x=[]
        self.batch_y =[]

    def reset(self):
      self.batch_x=[]
      self.batch_y =[]

    def flow_from_img(self, batch_size=1):
        while True:
          for x, y in zip(self.X, self.y):
            self.batch_x.append(x)
            self.batch_y.append(y)
            if len(self.batch_x)==batch_size:
              input_img=np.array(self.batch_x)
            if len(self.batch_y)==batch_size:
              input_y = np.array(self.batch_y)
              self.reset()
              yield input_img, input_y

inp=Input(batch_shape= (None, LENGTH_PER_UNIT, DIMENSION)) # (?, 48, 19200)
out1 = Bidirectional(LSTM(300, stateful=False, recurrent_dropout=0.3, return_sequences=True))(inp)
out2 = Bidirectional(LSTM(300, stateful=False, recurrent_dropout=0.3, return_sequences=True))(out1)
out3 = Flatten()(out2)
output = Dense(DIMENSION, activation='tanh')(out3)  # activation='linear'
model = Model(inp, output)
#model.load_weights("drive/My Drive/10_lstm.hdf5")
model.compile(optimizer=Adam(lr=0.0001), loss='mse')
model.summary()

"""
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 48, 8925)          0
_________________________________________________________________
bidirectional_1 (Bidirection (None, 48, 600)           22142400
_________________________________________________________________
bidirectional_2 (Bidirection (None, 48, 600)           2162400
_________________________________________________________________
flatten_1 (Flatten)          (None, 28800)             0
_________________________________________________________________
dense_1 (Dense)              (None, 8925)       257048925
"""

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config=config))


def step_decay(epoch):
    initial_lrate = 0.0001
    decay_rate = 0.5
    decay_steps = 8.0
    lrate = initial_lrate * math.pow(decay_rate,
           math.floor((1+epoch)/decay_steps))
    return lrate


callback=[]
#callback.append(HistoryCheckpoint(filepath='tb/LearningCurve_{history}.png', verbose=1, period=10))
callback.append(LearningRateScheduler(step_decay))
callback.append(ModelCheckpoint('{epoch:02d}_lstm.hdf5', monitor='loss', verbose=1))

train_gene=SequenceGenerator(X, Y)
tg=train_gene.flow_from_img()

val_gene=SequenceGenerator(Xv, yv)
vg=val_gene.flow_from_img()

history = model.fit_generator(tg, steps_per_epoch=1000, epochs=10, callbacks=callback,
                validation_data=vg, validation_steps=100)

predict = model.predict(X) #
reshape_img=predict.reshape(len(predict), 105, 85)
print(reshape_img.shape)  # (86, 105, 85)
