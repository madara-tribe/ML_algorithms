import tensorflow as tf
import os
import numpy as np
import cv2
from keras.preprocessing.image import array_to_img, img_to_array
from keras.optimizers import SGD
from keras import layers
from keras.preprocessing import image
from keras.models import Model
from keras.layers import *
from keras import backend as K
import matplotlib.pyplot as plt

from IPython.display import SVG
import pydot, graphviz
from tfrecord_classification.inception_resnet_v2 import InceptionResNetV2
  
  
 # grad_cam
def grad_cam(test_model, img, layer_name):
    # 前処理
    img=cv2.resize(img,(150,150))
    pre_input = np.expand_dims(img, axis=0)
    # 予測クラスの算出
    y_pred = test_model.predict(x=pre_input, batch_size=1)
    class_idx = np.argmax(y_pred,axis=1)
    class_output = test_model.output[:, class_idx]
    
    
    # 勾配の取得
    conv_output=test_model.get_layer(layer_name).output 
    grads=K.gradients(class_output, conv_output)[0]
    gradient_function = K.function([test_model.input], [conv_output, grads])
    output, grads_val = gradient_function([pre_input])
    output, grads_val = output[0], grads_val[0]
    
    # 重みを平均化してoutputに乗せる
    weights = np.mean(grads_val, axis=(0,1))
    cam = np.dot(output, weights)
    
    # 画像化してヒートマップに合成
    cam = cv2.resize(cam, (150, 150), cv2.INTER_LINEAR) 
    cam = np.maximum(cam, 0) 
    cam = cam / cam.max()
    jetcam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET) 
    
    # モノクロ画像に疑似的に色をつける
    jetcam = cv2.cvtColor(jetcam, cv2.COLOR_BGR2RGB)  # 色をRGBに変換
    jetcam = (np.float32(jetcam) + pre_input / 2) 
    jetcam=np.reshape(jetcam, (150,150,3))
    return array_to_img(jetcam)
  
  
  
# load model
test_model = InceptionResNetV2(include_top=True)
test_model.load_weights('inception_model_ep8.h5')
test_model.compile(optimizer=SGD(lr=0.01, momentum=0.9, decay=0.001, nesterov=True),
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])



# image
gucci_img=cv2.imread('images.jpeg')
gucci_img=cv2.resize(gucci_img,(150,150))
plt.imshow(gucci_img)
plt.show()

# plot
grad_cam(test_model,gucci_img,'mixed_7a')