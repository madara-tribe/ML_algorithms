import argparse
import os
import sys
import numpy as np
import cv2
from keras.callbacks import *
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.layers import *
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from models.cnn_model import create_model
from evaluate import evalation_by_plot_confusion_matrix, evaluate


def check_args(parsed_args):
    return parsed_args

def parse_args(args):
    parser     = argparse.ArgumentParser(description='param and images')
   
    parser.add_argument('--batch_size', help='batch size', default=10, type=int)
    parser.add_argument('--num_ep', help='num epochs', default=10, type=int)
    parser.add_argument('--weight_dir', help='folder path to save trained weight', default='weight_dir', type=str)
    parser.add_argument('--class_a_npy', help='numpuy train images1 path', type=str)
    parser.add_argument('--class_b_npy', help='numpuy train images2 path', type=str)
    parser.add_argument('--field_npy', help='numpuy evaluation images path', type=str)
    return check_args(parser.parse_args(args))


def load_img(npy_file, label_zeros=True):
  imgs = np.load(npy_file)
  num_cls = len(imgs)
  print(imgs.max(), imgs.min(), imgs.shape)
  if label_zeros:
    labels = np.zeros(num_cls)
  else:
    labels = np.ones(num_cls)
  return imgs, labels


def load_dataset(a_npy_file, b_npy_file):
    a_img, a_label = load_img(a_npy_file, True)
    b_img, b_label = load_img(b_npy_file, False)

    X = np.concatenate([a_img, b_img])
    y = np.concatenate([a_label, b_label])

    X_data = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))
    y_data = to_categorical(y)

    train_X, valid_X, train_y, valid_y = train_test_split(X_data, y_data, test_size = 0.2, shuffle=True)
    
    return train_X, valid_X, train_y, valid_y

def classifier(model, eval_npy):
    eval_img = np.load(eval_npy)
    eval_img = eval_img.reshape(len(eval_img), 40, 60, 1)
    y_pred = model.predict(eval_img)
    y_pred = np.argmax(y_pred, axis=1)
    num_a = 0
    num_b = 0
    for pred in y_pred:
        if pred==0:
            num_a += 1
        else:
            num_b += 1
    print('num class a is {}'.format(num_a))
    print('num class b is {}'.format(num_b))



def train(args=None, save_dir=None):
  if args is None:
    args = sys.argv[1:]
  args = parse_args(args)

  weight_dir = args.weight_dir
  if save_dir is None:
    os.mkdir(weight_dir)
  inputs = Input((40, 60, 1))
  model = create_model(inputs)
  
  train_X, valid_X, train_y, valid_y = load_dataset(args.class_a_npy, args.class_b_npy)

  # callback
  reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
  callback = [reduce_lr]
  
  # config
  batch_size = args.batch_size
  num_ep = args.num_ep

  # train
  try:
    model.fit(train_X, train_y, batch_size=batch_size, epochs=num_ep, callbacks=callback,
                          validation_data=(valid_X, valid_y), shuffle=True)
  finally:
    model.save(os.path.join(weight_dir, 'model_weight.h5'))
  
  print('sample validation test')
  evalation_by_plot_confusion_matrix(model, valid_X, valid_y)

  print('classifir and evaluate by alternative label')
  classifier(model, args.field_npy)
  evaluate(model, train_X, train_y, args.field_npy)


if __name__ == '__main__':
  train()
  
