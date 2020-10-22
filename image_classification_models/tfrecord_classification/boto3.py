import tensorflow as tf
from PIL import Image
import os
import sys
import cv2
import scipy.misc
from scipy.misc import imsave
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


df=pd.read_csv('/home/ubuntu/label_csvs/label_199.csv')
print(df.shape)
df.head()


# url size
url=list(df['image_url'])
len(url)

# get images from S3 to EC2

s3 = boto3.resource('s3', region_name='us-east-1')
bucket = s3.Bucket('image.buyma.com')   
image1=[]
no_object=[]
for i,v in enumerate(url):
    try:
        image1.append(mpimg.imread(io.BytesIO(bucket.Object(v).get()['Body'].read()), 'jp2'))
    except:
        no_object.append(i)


# image shape
print(len(image1))
img=[]
no_img=[]
for i,v in enumerate(image1):
    if v.ndim==3:
        img.append(v)
    if v.ndim==2:
        no_img.append(v)
img=[cv2.resize(i,(150,150)) for i in img]

imgs=[]
no_img=[]
for i in img:
    try:
        imgs.append(np.reshape(i,(150,150,3)))
    except ValueError:
        no_img.append(i)
imgs=np.reshape(imgs,(len(imgs),150,150, 3))
print(imgs.shape)


# devide image to train and test
train=imgs[500:]
test=imgs[:500]


flip_img=[]
for i in train:
    flip_img.append(cv2.flip(i, 1))
flip = np.r_[train, flip_img]
print(flip.shape)


# save train image to folder
for i,img in enumerate(flip):
    imsave('/home/ubuntu/train_dir/label_199/train_{:02d}.png'.format(i), img)


# save test image to folder
for i,img in enumerate(test):
    imsave('/home/ubuntu/test_dir/label_199/test_{:02d}.png'.format(i), img)

