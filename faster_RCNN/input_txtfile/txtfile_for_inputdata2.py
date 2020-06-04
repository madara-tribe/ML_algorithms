
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

anno_path = "drive/My Drive/dtc_train_annotations"
filename = [path for path in os.listdir(anno_path)]
sorted_file =[path for path in natsorted(filename)]

save_dir= "drive/My Drive/dtc_train1s"
Name, bbox, timeofday =[], [], []
for idx, file in enumerate(sorted_file):
  a = open(os.path.join(anno_path, file))
  Jso = json.load(a)
  name, _ = os.path.splitext(file)
  img_path = os.path.join(save_dir, name) + ".jpg"
  print(name, Jso['attributes']['timeofday'])
  for js in Jso['labels']:
    #print(img_path, js, Jso['attributes']['timeofday'])
    Name.append(img_path)
    bbox.append(js)
    timeofday.append(Jso['attributes']['timeofday'])
print(len(Name), len(bbox), len(timeofday))




df=[]
for idx, (img_path, BB, time) in enumerate(zip(Name, bbox, timeofday)):
  if time!='night':
    if '(1)' not in img_path:
      x1 = BB['box2d']['x1']
      y1 = BB['box2d']['y1']
      x2 = BB['box2d']['x2']
      y2 = BB['box2d']['y2']
      category = BB['category']
      if category=='Truck' or category=='Car' or category=='Signal':
        print(img_path, x1, y1, x2, y2, category)
        df.append([img_path, x1, y1, x2, y2, category])

with open('drive/My Drive/day_morning.txt', 'w+') as f:
    for d in df:
        f.write(','.join(map(str, d)) + '\n')



# plot bbox
def plt_bbox(img_path, BB, time):
    img = cv2.imread(img_path)
    copy_img = img.copy()
    x1 = BB['box2d']['x1']
    y1 = BB['box2d']['y1']
    x2 = BB['box2d']['x2']
    y2 = BB['box2d']['y2']
    category = BB['category']
    print(idx, category, time, [x1, y1, x2, y2])
    bbs = cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=5)
    plt.imshow(bbs),plt.show()
    bbox_parts = copy_img[y1:y2, x1:x2]
    plt.imshow(bbox_parts),plt.show()


for idx, (img_path, BB, time) in enumerate(zip(Name, bbox, timeofday)):
  plt_bbox(img_path, BB, time)
  if idx==10:
    break
