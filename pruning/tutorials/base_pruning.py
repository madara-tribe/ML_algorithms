from google.colab import drive
drive.mount('/content/drive')

!pip install -q tensorflow-model-optimization
!mkdir mnist_train

import tempfile
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras

%load_ext tensorboard


"""Just training MNIST"""
# Load MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the model architecture.
model = keras.Sequential([
  keras.layers.InputLayer(input_shape=(28, 28)),
  keras.layers.Reshape(target_shape=(28, 28, 1)),
  keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
  keras.layers.MaxPooling2D(pool_size=(2, 2)),
  keras.layers.Flatten(),
  keras.layers.Dense(10)
])

# Train the digit classification model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()
model.fit(
  train_images,
  train_labels,
  epochs=4,
  validation_split=0.1,
)

"""
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
reshape_1 (Reshape)          (None, 28, 28, 1)         0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 26, 26, 12)        120       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 13, 13, 12)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 2028)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 10)                20290     
=================================================================
Total params: 20,410
Trainable params: 20,410
Non-trainable params: 0
_________________________________________________________________
Epoch 1/4
1688/1688 [==============================] - 5s 3ms/step - loss: 0.2811 - accuracy: 0.9219 - val_loss: 0.1125 - val_accuracy: 0.9697
Epoch 2/4
1688/1688 [==============================] - 4s 3ms/step - loss: 0.1045 - accuracy: 0.9703 - val_loss: 0.0875 - val_accuracy: 0.9748
Epoch 3/4
1688/1688 [==============================] - 4s 3ms/step - loss: 0.0784 - accuracy: 0.9770 - val_loss: 0.0649 - val_accuracy: 0.9823
Epoch 4/4
1688/1688 [==============================] - 4s 3ms/step - loss: 0.0657 - accuracy: 0.9802 - val_loss: 0.0665 - val_accuracy: 0.9823
"""


"""evaluate test accuracy on base line and save results for after use"""
_, baseline_model_accuracy = model.evaluate(
    test_images, test_labels, verbose=0)

print('Baseline test accuracy:', baseline_model_accuracy)

_, keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model, keras_file, include_optimizer=False)
print('Saved baseline model to:', keras_file)

"""
Baseline test accuracy: 0.9796000123023987
Saved baseline model to: /tmp/tmpof8juhfd.h5
"""



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
batch_size = 128
epochs = 2
validation_split = 0.1 # 10% of training set will be used for validation set. 

num_images = train_images.shape[0] * (1 - validation_split)
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
model_for_pruning.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model_for_pruning.summary()


"""
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_model_optimization/python/core/sparsity/keras/pruning_wrapper.py:220: Layer.add_variable (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
Instructions for updating:
Please use `layer.add_weight` method instead.
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
prune_low_magnitude_reshape  (None, 28, 28, 1)         1         
_________________________________________________________________
prune_low_magnitude_conv2d ( (None, 26, 26, 12)        230       
_________________________________________________________________
prune_low_magnitude_max_pool (None, 13, 13, 12)        1         
_________________________________________________________________
prune_low_magnitude_flatten  (None, 2028)              1         
_________________________________________________________________
prune_low_magnitude_dense (P (None, 10)                40572     
=================================================================
Total params: 40,805
Trainable params: 20,410
Non-trainable params: 20,395
"""

"""モデルをベースラインに対してトレーニングおよび評価"""
logdir = tempfile.mkdtemp()

callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep(),
  tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]
  
model_for_pruning.fit(train_images, train_labels,
                  batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                  callbacks=callbacks)

"""
Epoch 1/2
  1/422 [..............................] - ETA: 0s - loss: 0.0700 - accuracy: 0.9844WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/summary_ops_v2.py:1277: stop (from tensorflow.python.eager.profiler) is deprecated and will be removed after 2020-07-01.
Instructions for updating:
use `tf.profiler.experimental.stop` instead.
WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0108s vs `on_train_batch_end` time: 0.0320s). Check your callbacks.
422/422 [==============================] - 3s 7ms/step - loss: 0.0827 - accuracy: 0.9773 - val_loss: 0.1029 - val_accuracy: 0.9738
Epoch 2/2
422/422 [==============================] - 3s 7ms/step - loss: 0.1030 - accuracy: 0.9717 - val_loss: 0.0882 - val_accuracy: 0.9773
"""

"""この例では、ベースラインと比較して、剪定後のテスト精度の損失は最小限です"""
_, model_for_pruning_accuracy = model_for_pruning.evaluate(
   test_images, test_labels, verbose=0)

print('Baseline test accuracy:', baseline_model_accuracy) 
print('Pruned test accuracy:', model_for_pruning_accuracy)

"""
Baseline test accuracy: 0.9796000123023987
Pruned test accuracy: 0.972599983215332
"""

%tensorboard --logdir={logdir}
#Reusing TensorBoard on port 6006 (pid 263), started 0:02:55 ago. (Use '!kill 263' to kill it.)
#<IPython.core.display.Javascript object>





