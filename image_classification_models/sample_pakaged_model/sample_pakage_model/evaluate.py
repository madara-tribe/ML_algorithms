import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import nmslib
import collections
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def plot_confusion_matrix(cm, classes, title=None, cmap=plt.cm.Blues):
  fig, ax = plt.subplots()
  im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
  ax.figure.colorbar(im, ax=ax)
  # We want to show all ticks...
  ax.set(xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes, yticklabels=classes,
        title=title,
        ylabel='True label',
        xlabel='Predicted label')

  # Rotate the tick labels and set their alignment.
  plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
          rotation_mode="anchor")

  # Loop over data dimensions and create text annotations.
  fmt = 'd'
  thresh = cm.max() / 2.
  for i in range(cm.shape[0]):
      for j in range(cm.shape[1]):
          ax.text(j, i, format(cm[i, j], fmt),
                  ha="center", va="center",
                  color="white" if cm[i, j] > thresh else "black")
  fig.tight_layout()
  return ax

def evalation_by_plot_confusion_matrix(model, eval_img, y_true):
  y_pred = model.predict(eval_img)
  y_pred = np.argmax(y_pred, axis=1)
  y_true = np.argmax(y_true, axis=1)

  cm = confusion_matrix(y_true, y_pred)
  plot_confusion_matrix(cm, classes=['0', '1'], title='Confusion matrix')
  # accuracy
  print('acuracy:{}'.format(accuracy_score(y_true, y_pred)))
  print(classification_report(y_true, y_pred))



def return_alternative_label(y_true, ids):
    return y_true[int(ids)]

def create_alternative_label(X, y, eval_img):
    Xs = X.reshape(len(X), -1)
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.addDataPointBatch(Xs)
    index.createIndex({'post': 2}, print_progress=True)

    print('create alternative label for eval')
    eval_imgs = eval_img.reshape(len(eval_img), -1)
    candidate = 1
    alternative_label = []
    for q in eval_imgs:
        ids, distances = index.knnQuery(q, k=candidate)
        alter_label = return_alternative_label(y, ids)
        alternative_label.append(alter_label)
    alternative_labels = np.array(alternative_label)
    return alternative_labels

def evaluate(model, X, y, eval_npy):
    print('evalation by alternative label')
    eval_img = np.load(eval_npy)
    alternative_label = create_alternative_label(X, y, eval_img)
    eval_imgs = eval_img.reshape(len(eval_img), 40, 60, 1)
    evalation_by_plot_confusion_matrix(model, eval_imgs, alternative_label)
