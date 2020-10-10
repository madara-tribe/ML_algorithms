from google.colab import drive
drive.mount('/content/drive')

!pip install -q tensorflow-model-optimization
!mkdir mnist_train

import tempfile
import os
import tensorflow as tf
import numpy as np

import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow.keras.preprocessing.sequence as sequence
from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from tensorflow import keras

%load_ext tensorboard

"""Just training MNIST"""
max_features = 20000
maxlen = 100  # cut texts after this number of words
batch_size = 32

print("Loading data...")
(x_train,
 y_train), (x_test,
            y_test) = keras.datasets.imdb.load_data(num_words=max_features)
print(len(x_train), "train sequences")
print(len(x_test), "test sequences")

print("Pad sequences (samples x time)")
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)

print("Build model...")
model = keras.models.Sequential()
model.add(keras.layers.Embedding(max_features, 128, input_length=maxlen))
model.add(keras.layers.LSTM(128))  # try using a GRU instead, for fun
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))
model.add(keras.layers.Activation("sigmoid"))

# try using different optimizers and different optimizer configs
model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])
model.summary()
model.fit(x_train, y_train, batch_size=batch_size, epochs=3,validation_data=(x_test, y_test))

"""
Loading data...
25000 train sequences
25000 test sequences
Pad sequences (samples x time)
x_train shape: (25000, 100)
x_test shape: (25000, 100)
Build model...
Model: "sequential_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_3 (Embedding)      (None, 100, 128)          2560000   
_________________________________________________________________
lstm_3 (LSTM)                (None, 128)               131584    
_________________________________________________________________
dropout_3 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 129       
_________________________________________________________________
activation_3 (Activation)    (None, 1)                 0         
=================================================================
Total params: 2,691,713
Trainable params: 2,691,713
Non-trainable params: 0
_________________________________________________________________
Epoch 1/3
782/782 [==============================] - 27s 34ms/step - loss: 0.4204 - accuracy: 0.8064 - val_loss: 0.3656 - val_accuracy: 0.8368
Epoch 2/3
782/782 [==============================] - 26s 33ms/step - loss: 0.2430 - accuracy: 0.9060 - val_loss: 0.4372 - val_accuracy: 0.8327
Epoch 3/3
782/782 [==============================] - 27s 34ms/step - loss: 0.1527 - accuracy: 0.9455 - val_loss: 0.4252 - val_accuracy: 0.8443
"""


"""evaluate test accuracy on base line and save results for after use"""
_, baseline_model_accuracy = model.evaluate(
    x_train, y_train, verbose=0)

print('Baseline test accuracy:', baseline_model_accuracy)

_, keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model, keras_file, include_optimizer=False)
print('Saved baseline model to:', keras_file)

#Baseline test accuracy: 0.9769200086593628
#Saved baseline model to: /tmp/tmpmhczug5e.h5


"""プルーニングを使用して事前トレーニング済みモデルを微調整する"""
"""adjust trained model by pruning"""

"""
モデル全体に​​枝刈りを適用し、モデルの概要でこれを確認します。

この例では、50％のスパース性（50％の重みでゼロ）でモデルを開始し、80％のスパース性で終了します。

包括的なガイドで 、モデルの精度を向上させるために一部のレイヤーを削除する方法を確認できます。

"""


import tensorflow_model_optimization as tfmot

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# Compute end step to finish pruning after 2 epochs.
batch_size = 32
epochs = 3
validation_split = 0.1 # 10% of training set will be used for validation set. 

num_images = x_train.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

# Define model for pruning.
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=end_step)
}

model_for_pruning = prune_low_magnitude(model, **pruning_params)

# `prune_low_magnitude` requires a recompile.
model_for_pruning.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model_for_pruning.summary()

"""
Model: "sequential_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
prune_low_magnitude_embeddin (None, 100, 128)          5120002   
_________________________________________________________________
prune_low_magnitude_lstm_3 ( (None, 128)               262659    
_________________________________________________________________
prune_low_magnitude_dropout_ (None, 128)               1         
_________________________________________________________________
prune_low_magnitude_dense_3  (None, 1)                 259       
_________________________________________________________________
prune_low_magnitude_activati (None, 1)                 1         
=================================================================
Total params: 5,382,922
Trainable params: 2,691,713
Non-trainable params: 2,691,209
"""

"""モデルをベースラインに対してトレーニングおよび評価"""
logdir = tempfile.mkdtemp()

callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep(),
  tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]
  
model_for_pruning.fit(x_train, y_train,
                  batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                  callbacks=callbacks)

"""
Epoch 1/3
  1/704 [..............................] - ETA: 0s - loss: 0.1394 - accuracy: 1.0000WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/summary_ops_v2.py:1277: stop (from tensorflow.python.eager.profiler) is deprecated and will be removed after 2020-07-01.
Instructions for updating:
use `tf.profiler.experimental.stop` instead.
704/704 [==============================] - 31s 45ms/step - loss: 0.0887 - accuracy: 0.9714 - val_loss: 0.1041 - val_accuracy: 0.9604
Epoch 2/3
704/704 [==============================] - 31s 44ms/step - loss: 0.0493 - accuracy: 0.9852 - val_loss: 0.1748 - val_accuracy: 0.9428
Epoch 3/3
704/704 [==============================] - 31s 44ms/step - loss: 0.0253 - accuracy: 0.9932 - val_loss: 0.2033 - val_accuracy: 0.9480
"""


"""この例では、ベースラインと比較して、剪定後のテスト精度の損失は最小限です"""
_, model_for_pruning_accuracy = model_for_pruning.evaluate(
   x_train, y_train, verbose=0)

print('Baseline test accuracy:', baseline_model_accuracy) 
print('Pruned test accuracy:', model_for_pruning_accuracy)
#Baseline test accuracy: 0.9769200086593628
#Pruned test accuracy: 0.9927999973297119


%tensorboard --logdir={logdir}
#<IPython.core.display.Javascript object>
