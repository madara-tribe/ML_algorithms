from google.colab import drive
drive.mount('/content/drive')

import os
import cv2
import numpy as np
from optparse import OptionParser
import time
import config
import matplotlib
import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import Input
from keras.models import Model
import keras_frcnn.roi_helpers
from keras_frcnn.visualize import draw_boxes_and_label_on_image_cv2
from keras_frcnn.simple_parser import get_data
C = config.Config()
C.network == 'resnet50'
import backbone.resnet as nn


def format_img_size(img, C):
    """ formats the image size based on config """
    img_min_side = float(C.im_size)
    (height, width ,_) = img.shape

    if width <= height:
        ratio = img_min_side/width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side/height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio

def format_img_channels(img, C):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def preprocessing_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):

    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return (real_x1, real_y1, real_x2 ,real_y2)


input_txt = 'drive/My Drive/voc.txt'
_, classes_count, class_mapping = get_data(input_txt)

if 'bg' not in class_mapping:
    class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
C.num_rois = 32

#if C.network == 'resnet50':
num_features = 1024

input_shape_img = (None, None, 3)
input_shape_features = (None, None, num_features)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)

weight_path = 'drive/My Drive'
base_net_weights = nn.get_weight_path(weight_path)
print('Loading weights from {}'.format(base_net_weights))
model_rpn.load_weights(base_net_weights, by_name=True)
model_classifier.load_weights(base_net_weights, by_name=True)

all_model_weights = 'drive/My Drive/model_frcnn_ep53.hdf5'
print('loading frcnn model from', all_model_weights)
model_rpn.load_weights(all_model_weights, by_name=True)
model_classifier.load_weights(all_model_weights, by_name=True)

model_rpn.compile(optimizer='adam', loss='mse')
model_classifier.compile(optimizer='adam', loss='mse')

def predict_single_image(img_path, model_rpn, model_classifier_only, cfg, class_mapping):
    img = cv2.imread(img_path)
    if img is None:
        print('reading image failed.')
        exit(0)
    plt.imshow(img),plt.show()
    X, ratio = preprocessing_img(img, cfg)
    if K.common.image_dim_ordering() == 'tf':
        X = np.transpose(X, (0, 2, 3, 1))
        print(X.shape)
    # get the feature maps and output from the RPN
    [Y1, Y2, F] = model_rpn.predict(X)
    # this is result contains all boxes, which is [x1, y1, x2, y2]
    result = roi_helpers.rpn_to_roi(Y1, Y2, cfg, K.common.image_dim_ordering(), overlap_thresh=0.7)
    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    result[:, 2] -= result[:, 0]
    result[:, 3] -= result[:, 1]
    bbox_threshold = 0.2

    # apply the spatial pyramid pooling to the proposed regions
    boxes = dict()
    for jk in range(result.shape[0] // cfg.num_rois + 1):
        rois = np.expand_dims(result[cfg.num_rois * jk:cfg.num_rois * (jk + 1), :], axis=0)
        if rois.shape[1] == 0:
            break
        if jk == result.shape[0] // cfg.num_rois:
            # pad R
            curr_shape = rois.shape
            target_shape = (curr_shape[0], cfg.num_rois, curr_shape[2])
            rois_padded = np.zeros(target_shape).astype(rois.dtype)
            rois_padded[:, :curr_shape[1], :] = rois
            rois_padded[0, curr_shape[1]:, :] = rois[0, 0, :]
            rois = rois_padded
        [p_cls, p_regr] = model_classifier_only.predict([F, rois])

        for ii in range(p_cls.shape[1]):
            if np.max(p_cls[0, ii, :]) < bbox_threshold or np.argmax(p_cls[0, ii, :]) == (p_cls.shape[2] - 1):
                continue
            cls_num = np.argmax(p_cls[0, ii, :])
            if cls_num not in boxes.keys():
                boxes[cls_num] = []
            (x, y, w, h) = rois[0, ii, :]
            try:
                (tx, ty, tw, th) = p_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                tx /= cfg.classifier_regr_std[0]
                ty /= cfg.classifier_regr_std[1]
                tw /= cfg.classifier_regr_std[2]
                th /= cfg.classifier_regr_std[3]
                x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
            except Exception as e:
                print(e)
                pass
            boxes[cls_num].append(
                [cfg.rpn_stride * x, cfg.rpn_stride * y, cfg.rpn_stride * (x + w), cfg.rpn_stride * (y + h),
                 np.max(p_cls[0, ii, :])])
    # add some nms to reduce many boxes
    print(boxes)
    for cls_num, box in boxes.items():
        boxes_nms = roi_helpers.non_max_suppression_fast(box, overlap_thresh=0.5)
        boxes[cls_num] = boxes_nms
        print(class_mapping[cls_num] + ":")
        for b in boxes_nms:
            b[0], b[1], b[2], b[3] = get_real_coordinates(ratio, b[0], b[1], b[2], b[3])
            print('{} prob: {}'.format(b[0: 4], b[-1]))
    img = draw_boxes_and_label_on_image_cv2(img, class_mapping, boxes)
    plt.imshow(img),plt.show()

dir_path = 'drive/My Drive/JPEGImages/2007_001457.jpg'
predict_single_image(dir_path, model_rpn, model_classifier, C, class_mapping)

