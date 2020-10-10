from google.colab import drive
drive.mount('/content/drive')

!pip install tensorflow_model_optimization
!mkdir mnist_train

from absl import app as absl_app
from absl import flags
import tensorflow as tf
from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule

ConstantSparsity = pruning_schedule.ConstantSparsity
keras = tf.keras
l = keras.layers

FLAGS = flags.FLAGS

batch_size = 128
num_classes = 10
epochs = 12
output_dir = 'mnist_train'


def build_sequential_model(input_shape):
  return tf.keras.Sequential([
      l.Conv2D(
          32, 5, padding='same', activation='relu', input_shape=input_shape),
      l.MaxPooling2D((2, 2), (2, 2), padding='same'),
      l.BatchNormalization(),
      l.Conv2D(64, 5, padding='same', activation='relu'),
      l.MaxPooling2D((2, 2), (2, 2), padding='same'),
      l.Flatten(),
      l.Dense(1024, activation='relu'),
      l.Dropout(0.4),
      l.Dense(num_classes, activation='softmax')
  ])


def build_functional_model(input_shape):
  inp = tf.keras.Input(shape=input_shape)
  x = l.Conv2D(32, 5, padding='same', activation='relu')(inp)
  x = l.MaxPooling2D((2, 2), (2, 2), padding='same')(x)
  x = l.BatchNormalization()(x)
  x = l.Conv2D(64, 5, padding='same', activation='relu')(x)
  x = l.MaxPooling2D((2, 2), (2, 2), padding='same')(x)
  x = l.Flatten()(x)
  x = l.Dense(1024, activation='relu')(x)
  x = l.Dropout(0.4)(x)
  out = l.Dense(num_classes, activation='softmax')(x)

  return tf.keras.models.Model([inp], [out])


def build_layerwise_model(input_shape, **pruning_params):
  return tf.keras.Sequential([
      prune.prune_low_magnitude(
          l.Conv2D(32, 5, padding='same', activation='relu'),
          input_shape=input_shape,
          **pruning_params),
      l.MaxPooling2D((2, 2), (2, 2), padding='same'),
      l.BatchNormalization(),
      prune.prune_low_magnitude(
          l.Conv2D(64, 5, padding='same', activation='relu'), **pruning_params),
      l.MaxPooling2D((2, 2), (2, 2), padding='same'),
      l.Flatten(),
      prune.prune_low_magnitude(
          l.Dense(1024, activation='relu'), **pruning_params),
      l.Dropout(0.4),
      prune.prune_low_magnitude(
          l.Dense(num_classes, activation='softmax'), **pruning_params)
  ])


def train_and_save(models, x_train, y_train, x_test, y_test, output_dir):
  for model in models:
    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer='adam',
        metrics=['accuracy'])

    # Print the model summary.
    model.summary()

    # Add a pruning step callback to peg the pruning step to the optimizer's
    # step. Also add a callback to add pruning summaries to tensorboard
    callbacks = [
        pruning_callbacks.UpdatePruningStep(),
        pruning_callbacks.PruningSummaries(log_dir=output_dir)
    ]

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Export and import the model. Check that accuracy persists.
    saved_model_dir = '/tmp/saved_model'
    print('Saving model to: ', saved_model_dir)
    tf.keras.models.save_model(model, saved_model_dir, save_format='tf')
    print('Loading model from: ', saved_model_dir)
    loaded_model = tf.keras.models.load_model(saved_model_dir)

    score = loaded_model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

if tf.keras.backend.image_data_format() == 'channels_first':
  x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
  x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
  input_shape = (1, img_rows, img_cols)
else:
  x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
  x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
  input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

"""
x_train shape: (60000, 28, 28, 1)
60000 train samples
10000 test samples
"""

pruning_params = {
      'pruning_schedule': ConstantSparsity(0.75, begin_step=2000, frequency=100)}
layerwise_model = build_layerwise_model(input_shape, **pruning_params)
layerwise_model.summary()

"""
Model: "sequential_4"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
prune_low_magnitude_conv2d_1 (None, 28, 28, 32)        1634      
_________________________________________________________________
max_pooling2d_10 (MaxPooling (None, 14, 14, 32)        0         
_________________________________________________________________
batch_normalization_5 (Batch (None, 14, 14, 32)        128       
_________________________________________________________________
prune_low_magnitude_conv2d_1 (None, 14, 14, 64)        102466    
_________________________________________________________________
max_pooling2d_11 (MaxPooling (None, 7, 7, 64)          0         
_________________________________________________________________
flatten_5 (Flatten)          (None, 3136)              0         
_________________________________________________________________
prune_low_magnitude_dense_10 (None, 1024)              6423554   
_________________________________________________________________
dropout_5 (Dropout)          (None, 1024)              0         
_________________________________________________________________
prune_low_magnitude_dense_11 (None, 10)                20492     
=================================================================
Total params: 6,548,274
Trainable params: 3,274,698
Non-trainable params: 3,273,576
"""

sequential_model = build_sequential_model(input_shape)
sequential_model.summary()

"""
Model: "sequential_5"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_12 (Conv2D)           (None, 28, 28, 32)        832       
_________________________________________________________________
max_pooling2d_12 (MaxPooling (None, 14, 14, 32)        0         
_________________________________________________________________
batch_normalization_6 (Batch (None, 14, 14, 32)        128       
_________________________________________________________________
conv2d_13 (Conv2D)           (None, 14, 14, 64)        51264     
_________________________________________________________________
max_pooling2d_13 (MaxPooling (None, 7, 7, 64)          0         
_________________________________________________________________
flatten_6 (Flatten)          (None, 3136)              0         
_________________________________________________________________
dense_12 (Dense)             (None, 1024)              3212288   
_________________________________________________________________
dropout_6 (Dropout)          (None, 1024)              0         
_________________________________________________________________
dense_13 (Dense)             (None, 10)                10250     
=================================================================
Total params: 3,274,762
Trainable params: 3,274,698
Non-trainable params: 64
"""

sequential_model = prune.prune_low_magnitude(
      sequential_model, **pruning_params)
sequential_model.summary()

"""
Model: "sequential_5"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
prune_low_magnitude_conv2d_1 (None, 28, 28, 32)        1634      
_________________________________________________________________
prune_low_magnitude_max_pool (None, 14, 14, 32)        1         
_________________________________________________________________
prune_low_magnitude_batch_no (None, 14, 14, 32)        129       
_________________________________________________________________
prune_low_magnitude_conv2d_1 (None, 14, 14, 64)        102466    
_________________________________________________________________
prune_low_magnitude_max_pool (None, 7, 7, 64)          1         
_________________________________________________________________
prune_low_magnitude_flatten_ (None, 3136)              1         
_________________________________________________________________
prune_low_magnitude_dense_12 (None, 1024)              6423554   
_________________________________________________________________
prune_low_magnitude_dropout_ (None, 1024)              1         
_________________________________________________________________
prune_low_magnitude_dense_13 (None, 10)                20492     
=================================================================
Total params: 6,548,279
Trainable params: 3,274,698
Non-trainable params: 3,273,581
"""

functional_model = build_functional_model(input_shape)
functional_model.summary()

"""
Model: "functional_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         [(None, 28, 28, 1)]       0         
_________________________________________________________________
conv2d_14 (Conv2D)           (None, 28, 28, 32)        832       
_________________________________________________________________
max_pooling2d_14 (MaxPooling (None, 14, 14, 32)        0         
_________________________________________________________________
batch_normalization_7 (Batch (None, 14, 14, 32)        128       
_________________________________________________________________
conv2d_15 (Conv2D)           (None, 14, 14, 64)        51264     
_________________________________________________________________
max_pooling2d_15 (MaxPooling (None, 7, 7, 64)          0         
_________________________________________________________________
flatten_7 (Flatten)          (None, 3136)              0         
_________________________________________________________________
dense_14 (Dense)             (None, 1024)              3212288   
_________________________________________________________________
dropout_7 (Dropout)          (None, 1024)              0         
_________________________________________________________________
dense_15 (Dense)             (None, 10)                10250     
=================================================================
Total params: 3,274,762
Trainable params: 3,274,698
Non-trainable params: 64
"""

functional_model = prune.prune_low_magnitude(
      functional_model, **pruning_params)
functional_model.summary()

"""
Model: "functional_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         [(None, 28, 28, 1)]       0         
_________________________________________________________________
prune_low_magnitude_conv2d_1 (None, 28, 28, 32)        1634      
_________________________________________________________________
prune_low_magnitude_max_pool (None, 14, 14, 32)        1         
_________________________________________________________________
prune_low_magnitude_batch_no (None, 14, 14, 32)        129       
_________________________________________________________________
prune_low_magnitude_conv2d_1 (None, 14, 14, 64)        102466    
_________________________________________________________________
prune_low_magnitude_max_pool (None, 7, 7, 64)          1         
_________________________________________________________________
prune_low_magnitude_flatten_ (None, 3136)              1         
_________________________________________________________________
prune_low_magnitude_dense_14 (None, 1024)              6423554   
_________________________________________________________________
prune_low_magnitude_dropout_ (None, 1024)              1         
_________________________________________________________________
prune_low_magnitude_dense_15 (None, 10)                20492     
=================================================================
Total params: 6,548,279
Trainable params: 3,274,698
Non-trainable params: 3,273,581
"""


models = [layerwise_model, sequential_model, functional_model]
train_and_save(models, x_train, y_train, x_test, y_test, output_dir)

"""
Model: "sequential_4"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
prune_low_magnitude_conv2d_1 (None, 28, 28, 32)        1634      
_________________________________________________________________
max_pooling2d_10 (MaxPooling (None, 14, 14, 32)        0         
_________________________________________________________________
batch_normalization_5 (Batch (None, 14, 14, 32)        128       
_________________________________________________________________
prune_low_magnitude_conv2d_1 (None, 14, 14, 64)        102466    
_________________________________________________________________
max_pooling2d_11 (MaxPooling (None, 7, 7, 64)          0         
_________________________________________________________________
flatten_5 (Flatten)          (None, 3136)              0         
_________________________________________________________________
prune_low_magnitude_dense_10 (None, 1024)              6423554   
_________________________________________________________________
dropout_5 (Dropout)          (None, 1024)              0         
_________________________________________________________________
prune_low_magnitude_dense_11 (None, 10)                20492     
=================================================================
Total params: 6,548,274
Trainable params: 3,274,698
Non-trainable params: 3,273,576
_________________________________________________________________
Epoch 1/12
  1/469 [..............................] - ETA: 0s - loss: 3.0613 - accuracy: 0.0938WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/summary_ops_v2.py:1277: stop (from tensorflow.python.eager.profiler) is deprecated and will be removed after 2020-07-01.
Instructions for updating:
use `tf.profiler.experimental.stop` instead.
  2/469 [..............................] - ETA: 14s - loss: 4.6921 - accuracy: 0.1211WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0131s vs `on_train_batch_end` time: 0.0433s). Check your callbacks.
469/469 [==============================] - 5s 10ms/step - loss: 0.1953 - accuracy: 0.9457 - val_loss: 0.1289 - val_accuracy: 0.9780
Epoch 2/12
469/469 [==============================] - 5s 10ms/step - loss: 0.0462 - accuracy: 0.9857 - val_loss: 0.0262 - val_accuracy: 0.9912
Epoch 3/12
469/469 [==============================] - 4s 10ms/step - loss: 0.0310 - accuracy: 0.9901 - val_loss: 0.0216 - val_accuracy: 0.9921
Epoch 4/12
469/469 [==============================] - 4s 10ms/step - loss: 0.0247 - accuracy: 0.9917 - val_loss: 0.0253 - val_accuracy: 0.9921
Epoch 5/12
469/469 [==============================] - 5s 10ms/step - loss: 0.0271 - accuracy: 0.9917 - val_loss: 0.0229 - val_accuracy: 0.9923
Epoch 6/12
469/469 [==============================] - 5s 10ms/step - loss: 0.0169 - accuracy: 0.9947 - val_loss: 0.0223 - val_accuracy: 0.9927
Epoch 7/12
469/469 [==============================] - 5s 10ms/step - loss: 0.0115 - accuracy: 0.9961 - val_loss: 0.0208 - val_accuracy: 0.9931
Epoch 8/12
469/469 [==============================] - 5s 10ms/step - loss: 0.0095 - accuracy: 0.9969 - val_loss: 0.0215 - val_accuracy: 0.9937
Epoch 9/12
469/469 [==============================] - 5s 10ms/step - loss: 0.0081 - accuracy: 0.9972 - val_loss: 0.0211 - val_accuracy: 0.9933
Epoch 10/12
469/469 [==============================] - 5s 10ms/step - loss: 0.0066 - accuracy: 0.9978 - val_loss: 0.0274 - val_accuracy: 0.9930
Epoch 11/12
469/469 [==============================] - 5s 10ms/step - loss: 0.0060 - accuracy: 0.9981 - val_loss: 0.0237 - val_accuracy: 0.9935
Epoch 12/12
469/469 [==============================] - 5s 10ms/step - loss: 0.0056 - accuracy: 0.9982 - val_loss: 0.0236 - val_accuracy: 0.9935
Test loss: 0.023552441969513893
Test accuracy: 0.9934999942779541
Saving model to:  /tmp/saved_model
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/training/tracking/tracking.py:111: Model.state_updates (from tensorflow.python.keras.engine.training) is deprecated and will be removed in a future version.
Instructions for updating:
This property should not be used in TensorFlow 2.0, as updates are applied automatically.
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/training/tracking/tracking.py:111: Layer.updates (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
Instructions for updating:
This property should not be used in TensorFlow 2.0, as updates are applied automatically.
INFO:tensorflow:Assets written to: /tmp/saved_model/assets
Loading model from:  /tmp/saved_model
Test loss: 0.02355252020061016
Test accuracy: 0.9934999942779541
Model: "sequential_5"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
prune_low_magnitude_conv2d_1 (None, 28, 28, 32)        1634      
_________________________________________________________________
prune_low_magnitude_max_pool (None, 14, 14, 32)        1         
_________________________________________________________________
prune_low_magnitude_batch_no (None, 14, 14, 32)        129       
_________________________________________________________________
prune_low_magnitude_conv2d_1 (None, 14, 14, 64)        102466    
_________________________________________________________________
prune_low_magnitude_max_pool (None, 7, 7, 64)          1         
_________________________________________________________________
prune_low_magnitude_flatten_ (None, 3136)              1         
_________________________________________________________________
prune_low_magnitude_dense_12 (None, 1024)              6423554   
_________________________________________________________________
prune_low_magnitude_dropout_ (None, 1024)              1         
_________________________________________________________________
prune_low_magnitude_dense_13 (None, 10)                20492     
=================================================================
Total params: 6,548,279
Trainable params: 3,274,698
Non-trainable params: 3,273,581
_________________________________________________________________
Epoch 1/12
  2/469 [..............................] - ETA: 29s - loss: 4.8833 - accuracy: 0.1328WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0112s vs `on_train_batch_end` time: 0.1108s). Check your callbacks.
469/469 [==============================] - 6s 13ms/step - loss: 0.2015 - accuracy: 0.9453 - val_loss: 0.1175 - val_accuracy: 0.9731
Epoch 2/12
469/469 [==============================] - 5s 11ms/step - loss: 0.0512 - accuracy: 0.9836 - val_loss: 0.0304 - val_accuracy: 0.9889
Epoch 3/12
469/469 [==============================] - 5s 11ms/step - loss: 0.0351 - accuracy: 0.9888 - val_loss: 0.0334 - val_accuracy: 0.9906
Epoch 4/12
469/469 [==============================] - 5s 12ms/step - loss: 0.0275 - accuracy: 0.9913 - val_loss: 0.0267 - val_accuracy: 0.9919
Epoch 5/12
469/469 [==============================] - 5s 12ms/step - loss: 0.0314 - accuracy: 0.9905 - val_loss: 0.0189 - val_accuracy: 0.9935
Epoch 6/12
469/469 [==============================] - 6s 12ms/step - loss: 0.0180 - accuracy: 0.9940 - val_loss: 0.0191 - val_accuracy: 0.9930
Epoch 7/12
469/469 [==============================] - 5s 12ms/step - loss: 0.0131 - accuracy: 0.9958 - val_loss: 0.0180 - val_accuracy: 0.9935
Epoch 8/12
469/469 [==============================] - 6s 12ms/step - loss: 0.0102 - accuracy: 0.9971 - val_loss: 0.0213 - val_accuracy: 0.9933
Epoch 9/12
469/469 [==============================] - 6s 12ms/step - loss: 0.0088 - accuracy: 0.9973 - val_loss: 0.0212 - val_accuracy: 0.9931
Epoch 10/12
469/469 [==============================] - 5s 12ms/step - loss: 0.0077 - accuracy: 0.9975 - val_loss: 0.0273 - val_accuracy: 0.9925
Epoch 11/12
469/469 [==============================] - 5s 12ms/step - loss: 0.0065 - accuracy: 0.9977 - val_loss: 0.0248 - val_accuracy: 0.9926
Epoch 12/12
469/469 [==============================] - 6s 12ms/step - loss: 0.0065 - accuracy: 0.9978 - val_loss: 0.0248 - val_accuracy: 0.9935
Test loss: 0.024762077257037163
Test accuracy: 0.9934999942779541
Saving model to:  /tmp/saved_model
INFO:tensorflow:Assets written to: /tmp/saved_model/assets
Loading model from:  /tmp/saved_model
Test loss: 0.02476215548813343
Test accuracy: 0.9934999942779541
Model: "functional_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         [(None, 28, 28, 1)]       0         
_________________________________________________________________
prune_low_magnitude_conv2d_1 (None, 28, 28, 32)        1634      
_________________________________________________________________
prune_low_magnitude_max_pool (None, 14, 14, 32)        1         
_________________________________________________________________
prune_low_magnitude_batch_no (None, 14, 14, 32)        129       
_________________________________________________________________
prune_low_magnitude_conv2d_1 (None, 14, 14, 64)        102466    
_________________________________________________________________
prune_low_magnitude_max_pool (None, 7, 7, 64)          1         
_________________________________________________________________
prune_low_magnitude_flatten_ (None, 3136)              1         
_________________________________________________________________
prune_low_magnitude_dense_14 (None, 1024)              6423554   
_________________________________________________________________
prune_low_magnitude_dropout_ (None, 1024)              1         
_________________________________________________________________
prune_low_magnitude_dense_15 (None, 10)                20492     
=================================================================
Total params: 6,548,279
Trainable params: 3,274,698
Non-trainable params: 3,273,581
_________________________________________________________________
Epoch 1/12
  2/469 [..............................] - ETA: 34s - loss: 4.6413 - accuracy: 0.1523WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0128s vs `on_train_batch_end` time: 0.1317s). Check your callbacks.
469/469 [==============================] - 6s 12ms/step - loss: 0.2072 - accuracy: 0.9453 - val_loss: 0.1000 - val_accuracy: 0.9827
Epoch 2/12
469/469 [==============================] - 5s 11ms/step - loss: 0.0487 - accuracy: 0.9848 - val_loss: 0.0416 - val_accuracy: 0.9856
Epoch 3/12
469/469 [==============================] - 5s 11ms/step - loss: 0.0319 - accuracy: 0.9895 - val_loss: 0.0276 - val_accuracy: 0.9912
Epoch 4/12
469/469 [==============================] - 5s 11ms/step - loss: 0.0263 - accuracy: 0.9922 - val_loss: 0.0207 - val_accuracy: 0.9938
Epoch 5/12
469/469 [==============================] - 6s 12ms/step - loss: 0.0384 - accuracy: 0.9885 - val_loss: 0.0230 - val_accuracy: 0.9921
Epoch 6/12
469/469 [==============================] - 6s 12ms/step - loss: 0.0193 - accuracy: 0.9940 - val_loss: 0.0242 - val_accuracy: 0.9928
Epoch 7/12
469/469 [==============================] - 5s 12ms/step - loss: 0.0135 - accuracy: 0.9956 - val_loss: 0.0196 - val_accuracy: 0.9940
Epoch 8/12
469/469 [==============================] - 6s 12ms/step - loss: 0.0114 - accuracy: 0.9959 - val_loss: 0.0206 - val_accuracy: 0.9936
Epoch 9/12
469/469 [==============================] - 6s 12ms/step - loss: 0.0093 - accuracy: 0.9970 - val_loss: 0.0193 - val_accuracy: 0.9930
Epoch 10/12
469/469 [==============================] - 5s 12ms/step - loss: 0.0069 - accuracy: 0.9977 - val_loss: 0.0222 - val_accuracy: 0.9941
Epoch 11/12
469/469 [==============================] - 6s 12ms/step - loss: 0.0072 - accuracy: 0.9978 - val_loss: 0.0214 - val_accuracy: 0.9939
Epoch 12/12
469/469 [==============================] - 6s 12ms/step - loss: 0.0062 - accuracy: 0.9981 - val_loss: 0.0241 - val_accuracy: 0.9934
Test loss: 0.024091007187962532
Test accuracy: 0.993399977684021
Saving model to:  /tmp/saved_model
INFO:tensorflow:Assets written to: /tmp/saved_model/assets
Loading model from:  /tmp/saved_model
Test loss: 0.02409108728170395
Test accuracy: 0.993399977684021
"""