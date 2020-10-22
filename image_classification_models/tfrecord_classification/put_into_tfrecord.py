import tensorflow as tf
from PIL import Image
import os
import sys
import cv2
import scipy.misc
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import boto3
import io
import collections
from collections import OrderedDict



# read train image from path  (ex: range(0, 13)=> 0~12)
path=[os.path.join('/home/ubuntu/train_dir/label_%d' % i) for i in range(0, 200)]


N=199 # label number
print(len(path))
pa=path[N]
pa


img=[]
for imgs in os.listdir(pa):
    img.append(cv2.imread(pa+'/'+imgs))
image=np.reshape(img, (len(img),150,150,3))
print('image shape:{}'.format(image.shape))
print('image:{}'.format(len(image)))

shape=len(image)
k=np.array([N]*shape)
label=np.reshape(k, (shape,))
print('label:{}'.format(label.shape))
label


listed=[]
for i, v in zip(image, label):
    listed.append([v,i])
print('listed:{}'.format(len(listed)))
      

tfrecord='/home/ubuntu/train_tf/record_%d.tfrecords'% N
tfrecord


# put into tfrecord
writer = tf.python_io.TFRecordWriter(tfrecord)
for label, img in listed:
    record = tf.train.Example(features=tf.train.Features(feature={
          "label": tf.train.Feature(
              int64_list=tf.train.Int64List(value=[label])),
          "image": tf.train.Feature(
              bytes_list=tf.train.BytesList(value=[img.tostring()]))
      }))
 
    writer.write(record.SerializeToString())
writer.close()

