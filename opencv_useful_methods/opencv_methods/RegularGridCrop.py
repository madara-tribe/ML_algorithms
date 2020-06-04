import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

# difine target image
target = cv2.imread('database/original.jpg')
plt.imshow(target)
plt.show()

##### draw grid #####
y_step=20 # height grid interval (pixel)
x_step=40 # width grid interval (pixel)

img_y,img_x=target.shape[:2]

# draw white side line(RGB==255) from y_step to img_y at each y_step
target[y_step:img_y:y_step, :, :] = 255
# draw white vertical line as above
target[:, x_step:img_x:x_step, :] = 255

plt.imshow(target)
plt.show()
