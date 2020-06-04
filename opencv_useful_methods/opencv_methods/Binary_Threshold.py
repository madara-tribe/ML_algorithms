import matplotlib.pyplot as plt
import os
import numpy as np
import cv2


# difine target image
target = cv2.imread('database/original.jpg')


def binary_threshold(img):
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    under_thresh = 105
    upper_thresh = 145
    maxValue = 255
    th, drop_back = cv2.threshold(grayed, under_thresh, maxValue, cv2.THRESH_BINARY)
    th, clarify_born = cv2.threshold(grayed, upper_thresh, maxValue, cv2.THRESH_BINARY_INV)
    merged = np.minimum(drop_back, clarify_born)
    plt.imshow(merged)
    plt.show()


binary_threshold(target)
