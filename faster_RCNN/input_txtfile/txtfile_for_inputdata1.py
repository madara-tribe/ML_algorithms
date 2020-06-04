
from google.colab import drive
drive.mount('/content/drive')

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from natsort import natsorted
import json
import glob
from skimage.feature import hog
import xml.etree.ElementTree as ET
from os import getcwd

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def convert_annotation(anno_path, in_file):
    tree=ET.parse(os.path.join(anno_path, in_file))
    root = tree.getroot()
    save_name = 'drive/My Drive/JPEGImages/' + root.find('filename').text
    fname = 'drive/My Drive/JPEGImages/' + root.find('filename').text
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        return fname, save_name, b, cls

anno_path = "drive/My Drive/Annotations"
fname = [path for path in os.listdir(anno_path)]
sorted_file =[path for path in natsorted(fname)]

df=[]
for idx, f in enumerate(sorted_file):
    fname, save,  b, cls = convert_annotation(anno_path, f)
    #print(fname, save, b, cls)
    img = cv2.imread(fname)
    if idx<10:
        bbox = cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), color=(0, 0, 255), thickness=10)
        plt.imshow(bbox),plt.show()
    print(save, b[0], b[1], b[2], b[3], cls)
    df.append([save, b[0], b[1], b[2], b[3], cls])
    if idx==500:
        break

# save
with open('drive/My Drive/voc.txt', 'w+') as f:
  for d in df:
    f.write(','.join(map(str, d)) + '\n')
