import tempfile
import os
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import backend, optimizers
import tensorflow.keras.preprocessing.sequence as sequence
from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import sys, time, warnings
import gc #DB
from config import fcn_config as cfg
from config import fcn8_cnn as cnn
from tqdm import tqdm
from unet import UNET_v2
warnings.simplefilter('ignore')



#%load_ext tensorboard


HEIGHT = 224
WIDTH  = 224 
print(HEIGHT, WIDTH)
N_CLASSES = 11

BATCH_SIZE = 4
EPOCHS = 3



model = UNET_v2(N_CLASSES, HEIGHT, WIDTH)
model.summary()

dir_train_img='X/cmo_train'
dir_train_seg='X/indexmap'
dir_valid_img = 'X/val_img'
dir_valid_seg = 'X/val_anno'

train_images = os.listdir(dir_train_img)
train_images.sort()
train_segmentations  = os.listdir(dir_train_seg)
train_segmentations.sort()
X_train, Y_train = [], []

for im , seg in tqdm(zip(train_images,train_segmentations)):
    X_train.append(cnn.NormalizeImageArr(os.path.join(dir_train_img,im), WIDTH, HEIGHT))
    Y_train.append(cnn.LoadSegmentationArr( os.path.join(dir_train_seg,seg) , N_CLASSES , WIDTH, HEIGHT))


X_train, Y_train = np.array(X_train), np.array(Y_train)
#X_train, Y_train = augmentation(X_train, Y_train)
#X_train2, Y_train2 = np.array(X_train2), np.array(Y_train2)

print(X_train.shape,Y_train.shape)
print(X_train.max(), X_train.min())


sgd = optimizers.SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])


import tensorflow_model_optimization as tfmot

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# Compute end step to finish pruning after 2 epochs.
validation_split = 0.1 # 10% of training set will be used for validation set. 
epochs=3
batch_size=4

num_images = X_train.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

# Define model for pruning.
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50, final_sparsity=0.80, begin_step=0, end_step=end_step)}

model_for_pruning = prune_low_magnitude(model, **pruning_params)

# `prune_low_magnitude` requires a recompile.
model_for_pruning.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model_for_pruning.summary()

"""train and evaluation by pruning model for base model as base line"""
logdir = 'output'

callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep(),
  tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]
  
#model_for_pruning.fit(X_train,Y_train,
 #                 batch_size=batch_size, epochs=epochs, validation_split=validation_split, callbacks=callbacks)


#_, model_for_pruning_accuracy = model_for_pruning.evaluate(
 #  X_train,Y_train, verbose=0)

#print('Baseline test accuracy:', baseline_model_accuracy) 
#print('Pruned test accuracy:', model_for_pruning_accuracy)
