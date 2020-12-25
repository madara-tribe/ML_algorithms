import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import sys, time, warnings
from datetime import datetime #DB
import gc #DB
from tqdm import tqdm
from random_erasing import augmentation
from data_loader import verify_segmentation_dataset
from config import fcn_config as cfg
from config import fcn8_cnn as cnn
from config import unet as unet
from sklearn.utils import shuffle
import pandas as pd

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
import tensorflow_model_optimization as tfmot


warnings.filterwarnings("ignore")

HEIGHT = 224
WIDTH  = 224 
print(HEIGHT, WIDTH)
N_CLASSES = 11
BATCH_SIZE = 4
EPOCHS = 3
Pruninglogdir = 'logdir'

model_type = 2
if (model_type==1):
        model = unet.UNET_v1(N_CLASSES, HEIGHT, WIDTH)
elif (model_type==2):
        model = unet.UNET_v2(N_CLASSES, HEIGHT, WIDTH)
else:
        model = unet.UNET_v3(N_CLASSES, HEIGHT, WIDTH)

model.summary()





dir_train_img='X/cmo_train'
dir_train_seg='X/indexmap'
dir_valid_img = 'X/val_img'
dir_valid_seg = 'X/val_anno'

def verify_datasets(validate=True):
    n_classes=11
    print("Verifying training dataset")
    verified, tr_len, _ = verify_segmentation_dataset(dir_train_img,
                                           dir_train_seg,
                                           n_classes)
    assert verified
    if validate:
        print("Verifying validation dataset")
        verified, val_len, _ = verify_segmentation_dataset(dir_valid_img,
                                               dir_valid_seg,
                                               n_classes)
        assert verified
#verify_datasets(validate=True)

# load training images
train_images = os.listdir(dir_train_img)
train_images.sort()
train_segmentations  = os.listdir(dir_train_seg)
train_segmentations.sort()
X_train, X_train2 = [], []
Y_train, Y_train2 = [], []

for im , seg in tqdm(zip(train_images,train_segmentations)):
    X_train.append(cnn.NormalizeImageArr(os.path.join(dir_train_img,im), WIDTH, HEIGHT))
    Y_train.append(cnn.LoadSegmentationArr( os.path.join(dir_train_seg,seg) , N_CLASSES , WIDTH, HEIGHT)  )


for im , seg in tqdm(zip(train_images,train_segmentations)):
    X_train.append(cnn.ContrastNormalizeImageArr(os.path.join(dir_train_img,im), WIDTH, HEIGHT, gamma=0.5))
    Y_train.append(cnn.LoadSegmentationArr( os.path.join(dir_train_seg,seg) , N_CLASSES , WIDTH, HEIGHT)  )


for im , seg in tqdm(zip(train_images,train_segmentations)):
    X_train.append(cnn.ContrastNormalizeImageArr(os.path.join(dir_train_img,im), WIDTH, HEIGHT, gamma=0.3))
    Y_train.append(cnn.LoadSegmentationArr( os.path.join(dir_train_seg,seg) , N_CLASSES , WIDTH, HEIGHT)  )


X_train, Y_train = np.array(X_train), np.array(Y_train)
#X_train, Y_train = augmentation(X_train, Y_train)
#X_train2, Y_train2 = np.array(X_train2), np.array(Y_train2)

print(X_train.shape,Y_train.shape)
print(X_train.max(), X_train.min())

# load validation images
valid_images = os.listdir(dir_valid_img)
valid_images.sort()
valid_segmentations  = os.listdir(dir_valid_seg)
valid_segmentations.sort()
X_valid, X_valid2 = [], []
Y_valid, Y_valid2 = [], []
for im , seg in tqdm(zip(valid_images,valid_segmentations)):
    X_valid.append( cnn.NormalizeImageArr(os.path.join(dir_valid_img,im), WIDTH, HEIGHT) )
    Y_valid.append( cnn.LoadSegmentationArr( os.path.join(dir_valid_seg,seg) , N_CLASSES , WIDTH, HEIGHT))

for im , seg in tqdm(zip(valid_images,valid_segmentations)):
    X_valid.append(cnn.ContrastNormalizeImageArr(os.path.join(dir_valid_img,im), WIDTH, HEIGHT, gamma=0.5) )
    Y_valid.append(cnn.LoadSegmentationArr( os.path.join(dir_valid_seg,seg) , N_CLASSES , WIDTH, HEIGHT)  )


for im , seg in tqdm(zip(valid_images,valid_segmentations)):
    X_valid.append(cnn.ContrastNormalizeImageArr(os.path.join(dir_valid_img,im), WIDTH, HEIGHT, gamma=0.3) )
    Y_valid.append(cnn.LoadSegmentationArr( os.path.join(dir_valid_seg,seg) , N_CLASSES , WIDTH, HEIGHT)  )

X_valid, Y_valid = np.array(X_valid) , np.array(Y_valid)
#X_valid, Y_valid = augmentation(X_valid, Y_valid)
#X_valid2, Y_valid2 = np.array(X_valid2) , np.array(Y_valid2)


print(X_valid.shape,Y_valid.shape)
print(X_valid.max(),X_valid.min())

checkpoint_path = "train_ck/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
 
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, period=1)

callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep(),
  tfmot.sparsity.keras.PruningSummaries(log_dir=Pruninglogdir),
  cp_callback
]

def pruning_model(model, X_train):
    batch_size = BATCH_SIZE
    epochs = EPOCHS
    # Compute end step to finish pruning after 2 epochs.
    validation_split = 0.1 # 10% of training set will be used for validation set.   
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    num_images = X_train.shape[0] * (1 - validation_split)
    end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

    # Define model for pruning.
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50, final_sparsity=0.80, begin_step=0, end_step=end_step)}

    model_for_pruning = prune_low_magnitude(model, **pruning_params)
    return model_for_pruning

model_for_pruning = pruning_model(model, X_train)
sgd = optimizers.SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)
model_for_pruning.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
model_for_pruning.summary()



print('start training')
startTime1 = datetime.now() #DB
hist1 = model_for_pruning.fit(X_train,Y_train, validation_data=(X_valid,Y_valid), batch_size=BATCH_SIZE,epochs=EPOCHS, callbacks = callbacks, verbose=2)
endTime1 = datetime.now()
diff1 = endTime1 - startTime1
print("\n")
print("Elapsed time for Keras training (s): ", diff1.total_seconds())
print("\n")

for key in ["loss", "val_loss"]:
    plt.plot(hist1.history[key],label=key)
plt.legend()

plt.savefig("keras_model/unet_model" + str(model_type) + "_training_curves_" + str(WIDTH) + "x" + str(HEIGHT) + ".png")

model_for_pruning.save("keras_model/ep" + str(EPOCHS) + "_trained_unet_model" + str(model_type) + "_" + str(WIDTH) + "x" + str(HEIGHT) + ".hdf5")
print("\nEnd of UNET training\n")


