import math
import sys
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.feature import hog
import cv2
from sklearn.svm import SVC
import joblib
import json

from rotation import rotation

# define parameters of HOG feature extraction
orientations = 8
pixels_per_cell = (16, 16)
cells_per_block = (1, 1)

def crop_query(query_path):
    img = cv2.imread(query_path)
    if img is not None:
        h, w = img.shape[:2]
        crop_size = int(abs(h-w)/2)
    #  crop to square
    if h>w:
        img=img[crop_size:-crop_size, :]
    else:
        img=img[:, crop_size:-crop_size]
    return img


def load_dataset(dir_path, resize_shape):
    trains=[]
    for im in os.listdir(dir_path):
        img = cv2.imread(dir_path+'/'+im)
        if img is not None:
            img = cv2.resize(img, (resize_shape, resize_shape))
            trains.append(img)
    return np.array(trains).reshape(len(trains), resize_shape, resize_shape, 3)

def train_svm_model(query, trains):
    X = []
    y = []
    print("train start")

    for classId, img in enumerate(trains):
        # 進捗表示
        sys.stdout.write(".")
        plt.imshow(img,),plt.show()
        print(img.shape)
        fd = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block)
        X.append(fd)
        y.append(classId)
    detector = SVC(gamma='auto')
    detector.fit(X, y)
    print("train finish")

    pd = hog(query, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block)
    _idx = detector.predict(pd.reshape(1, len(pd)))
    print('predict idx {} '.format(_idx))
    # load query and similar image
    sim_image = trains[int(_idx)]
    return sim_image, detector





def test_svm_model(query, tests, detector=True):
    print("test start")

    for classId, test_img in enumerate(tests):
        td = hog(test_img, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block)
        pedicted_idx = detector.predict(td.reshape(1, len(td)))

        if int(pedicted_idx)==classId:
            print("success!!")
        else:
            print("fail")

        print("predicted id: ", pedicted_idx, "class Id", classId )

    print("test finished")
    joblib.dump(detector, 'svm_detector.pkl', compress=True)
