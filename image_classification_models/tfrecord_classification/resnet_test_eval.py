import os
import math
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model, Model
from keras.utils import multi_gpu_model, print_summary
from keras import layers
from keras import backend as K
from sklearn.metrics import confusion_matrix as c_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, roc_curve
from inception_resnet_v2 import InceptionResNetV2

from tensorflow.python.client import device_lib
device_lib.list_local_devices()



# test image for input 
def distorted_input(data_files, batch_size, num_readers = 1):
    num_class=200
    
    filename_queue = tf.train.string_input_producer(data_files, shuffle=False, capacity=1)
    num_preprocess_threads = 4
    examples_per_shard = 1024
    min_queue_examples = examples_per_shard * 16
    examples_queue = tf.FIFOQueue(capacity=examples_per_shard + 3 * batch_size, 
                                      dtypes=[tf.string])

    # Create queue
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
    image.set_shape([height, width, 3])
    
    return image, label


def eval();
  # read tfrecord files for test
  path='/home/ubuntu/test_tf'
  filenames = [os.path.join(path, 'test_%d.tfrecords' % i) for i in range(0, 2)]
  for f in filenames:
      if not tf.gfile.Exists(f):
          raise ValueError('Failed to find file: ' + f)
  len(filenames)


  # test
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
  K.set_session(sess)

  NUM_GPUS=8
  batch_size=125*NUM_GPUS

  test_image, test_labels= distorted_input(filenames, batch_size)
  with tf.device('/cpu:0'):
      test_model = InceptionResNetV2(test_image)
  test_model.load_weights('/home/ubuntu/check_dir/inception_model_ep8.h5')
  test_model= multi_gpu_model(test_model, gpus=NUM_GPUS)
  test_model.compile(optimizer=SGD(lr=0.01, momentum=0.9, decay=0.001, nesterov=True),
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'], target_tensors=[test_labels])
  print_summary(test_model)


  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess, coord)

  _, acc = test_model.evaluate(x=None, verbose=1, steps=100)
  print('\nTest accuracy: {0}'.format(acc))

  y_pred = test_model.predict(x=None,verbose=1,steps=100)
  LABEL=[]
  for i in range(100):
      LABEL.append(sess.run(test_labels))
  top_k=sess.run(tf.nn.top_k(y_pred,k=3,sorted=False))
  coord.request_stop()
  coord.join(threads)




  # form shape
  label=[]
  for l in LABEL:
      for i in l:
          label.append(i)
  label=np.array(label)
  print(label.shape)


  label_batch=np.argmax(label, axis=1)
  print(label_batch.shape)

  f_pred=np.argmax(y_pred, axis=1)
  print(f_pred.shape)

  cm=c_matrix(label_batch,f_pred)
  print(cm.shape)

  # confusion matrix
  print(pd.DataFrame(cm))


  norm_conf = []
  for i in cm:
      a = 0
      tmp_arr = []
      a = sum(i, 0)
      for j in i:
          tmp_arr.append(float(j)/float(a))
      norm_conf.append(tmp_arr)

  fig = plt.figure()
  plt.clf()
  ax = fig.add_subplot(111)
  ax.set_aspect(1)
  res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                  interpolation='nearest')

  width, height = cm.shape

  for x in range(width):
      for y in range(height):
          ax.annotate(str(cm[x][y]), xy=(y, x),
                      horizontalalignment='center',
                      verticalalignment='center')

  cb = fig.colorbar(res)
  label_string = ['{}'.format(i) for i in range(200)]
  plt.xticks(range(width), label_string[:width])
  plt.yticks(range(height), label_string[:height])


  # In[26]:


  # Accuracy , F-score.etc
  print('acuracy:{}'.format(accuracy_score(label_batch,f_pred)))
  label_string = ['{}'.format(i) for i in range(200)]
  print(classification_report(label_batch, f_pred,target_names=label_string))

