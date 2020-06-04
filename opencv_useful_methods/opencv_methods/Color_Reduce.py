import sys
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2


# difine target image
target = cv2.imread('database/original.jpg')



def reduce_color(target):
    Z = target.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    K = 2 # amount of colors
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((target.shape))

    plt.imshow(res2)
    plt.show()
    print(res2.shape)
