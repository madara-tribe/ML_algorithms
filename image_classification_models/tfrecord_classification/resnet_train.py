import os
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import print_summary
from keras.utils import multi_gpu_model
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import History, LearningRateScheduler, Callback
from keras import layers
from keras.models import Model, save_model
from keras.layers import Activation, Concatenate, AveragePooling2D, BatchNormalization
from keras.layers import Conv2D, Dense, Input, Lambda, GlobalMaxPooling2D, MaxPooling2D, GlobalAveragePooling2D
from keras import backend as K
from inception_resnet_v2 import InceptionResNetV2
tf.logging.set_verbosity(tf.logging.ERROR)


from tensorflow.python.client import device_lib
device_lib.list_local_devices()


# image input
def distorted_input(data_files, batch_size, train=True, num_readers = 60):
    num_class=200
    if train:
        filename_queue = tf.train.string_input_producer(data_files, shuffle=True, capacity=16)
    else:
        filename_queue = tf.train.string_input_producer(data_files, shuffle=False, capacity=1)
    num_preprocess_threads = 60
    examples_per_shard = 1024
    min_queue_examples = examples_per_shard * 16
    if train:
        examples_queue = tf.RandomShuffleQueue(capacity=min_queue_examples + 3 * batch_size,
                min_after_dequeue=min_queue_examples, dtypes=[tf.string])
    else:
        examples_queue = tf.FIFOQueue(capacity=examples_per_shard + 3 * batch_size, 
                                      dtypes=[tf.string])

    # Create queue
    if num_readers > 1:
        enqueue_ops = []
        for _ in range(num_readers):
            reader = tf.TFRecordReader()
            _, value = reader.read(filename_queue)
            enqueue_ops.append(examples_queue.enqueue([value]))
        tf.train.queue_runner.add_queue_runner(
            tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
        example_serialized = examples_queue.dequeue()
    else:
        reader = tf.TFRecordReader()
        _, example_serialized = reader.read(filename_queue)
        
    images_and_labels = []
    for thread_id in range(num_preprocess_threads):
        image, label_index = parse_example_proto(example_serialized)
        images_and_labels.append([image, label_index])
    
    images, label_index_batch = tf.train.batch_join(images_and_labels,
             batch_size=batch_size, capacity=2 * num_preprocess_threads * batch_size)
    
    height = 150
    width = 150
    images = tf.reshape(images, shape=[batch_size, height, width, 3])
    
    return tf.subtract(tf.div(images,127.5), 1.0), tf.one_hot(tf.reshape(label_index_batch, [batch_size]), num_class)


def parse_example_proto(serialized_example):
    height = 150
    width = 150
    features = tf.parse_single_example(serialized_example,
                        features={"label": tf.FixedLenFeature([], tf.int64),
                                  "image": tf.FixedLenFeature([], tf.string)})
    label = tf.cast(features["label"], tf.int32)
    imgin = tf.reshape(tf.decode_raw(features["image"], tf.uint8), tf.stack([height, width, 3]))
    image = tf.cast(imgin, tf.float32)
    distorted_image = tf.image.random_flip_left_right(image)
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    distorted_image.set_shape([height, width, 3])
    return distorted_image, label



def train():
  # read tfrecord files
  path='/home/ubuntu/train_tf'
  filenames = [os.path.join(path, 'record_%d.tfrecords' % i) for i in range(0, 200)]
  for f in filenames:
      if not tf.gfile.Exists(f):
          raise ValueError('Failed to find file: ' + f)
  print(len(filenames))


  # train
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
  K.set_session(sess)

  # lr = tf.train.exponential_decay(0.1, global_step, decay_steps, learning_rate_decay_factor, staircase=True)

  def step_decay(epoch):
      initial_lrate = 0.01
      decay_rate = 0.5
      decay_steps = 8.0
      lrate = initial_lrate * math.pow(decay_rate,  
             math.floor((1+epoch)/decay_steps))
      return lrate


  NUM_GPUS = 8
  batch_size=32*NUM_GPUS

  train_image, train_labels=distorted_input(filenames, batch_size, train=True)

  with tf.device('/cpu:0'):
      train_model= InceptionResNetV2(train_image)
  pmodel= multi_gpu_model(train_model, gpus=NUM_GPUS)
  pmodel.compile(optimizer=SGD(lr=0.01, momentum=0.9, decay=0.001, nesterov=True),
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'], target_tensors=[train_labels])

  # callback
  history = History()
  callback=[]
  callback.append(history)
  callback.append(LearningRateScheduler(step_decay))


  tf.train.start_queue_runners(sess=sess)
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess, coord)
  try:
      pmodel.fit(steps_per_epoch=int(5101031/batch_size), epochs=3, callbacks=callback)
  finally:
      train_model.save('/home/ubuntu/check_dir/inception_model_ep8.h5')
  coord.request_stop()
  coord.join(threads)

  K.clear_session()

