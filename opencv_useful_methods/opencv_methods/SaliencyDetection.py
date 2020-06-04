# 顕著点extract

import sys
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage import data, color, exposure
from skimage.feature import hog
import cv2


# difine target image
target = cv2.imread('database/original.jpg')
plt.imshow(target)
plt.show()

# better way
saliency = cv2.saliency.StaticSaliencyFineGrained_create()
(success, saliencyMap) = saliency.computeSaliency(target)

if success is True:
    plt.imshow(saliencyMap)
    plt.gray()
    plt.show()
