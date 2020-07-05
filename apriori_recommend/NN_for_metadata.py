import numpy as np
import pandas as pd
import seaborn as sns
import collections 
from sklearn import datasets, metrics, cross_validation
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.cross_validation import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as c_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, roc_curve
import tensorflow as tf
from keras.utils import print_summary
from keras import layers
from keras.models import Model, save_model, Sequential
from keras.layers import Activation, Conv2D, Dense, Input, Lambda,
from keras import backend as K
get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 12, 4

df=pd.read_csv('cate_meta.csv')
df=df.drop('cate_name3', axis=1)
df=df[df['cate_id2']>1000]
df=df.fillna(0)
print(df.shape)

df.head()



# make 'mata' row

meta={}
for i,v in df.iterrows():
    meta.setdefault(int(v['cate_id2']), []).append(int(v['cate_id']))
m=[[i,v] for i,v in enumerate(meta)]

meta_label=[]
for label in m:
    for ids, v in df.iterrows():
        if v['cate_id2'] == label[1]:
            meta_label.append(label[0])
df['meta']=[i for i in meta_label]


# increase data
X=df
X=pd.concat([X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X])
X=pd.concat([X,X,X])
X.shape


# train　データ

# ラベル
train_y=np.array(X['meta'])
train_y=np.reshape(train_y, (len(train_y),))            # クラス分類を整数値のベクトルで表現したもの
n_labels = len(np.unique(train_y))  # 分類クラスの数 = 5
train_y=np.eye(n_labels)[train_y]
train_y=np.reshape(train_y, (len(train_y), 91))
print(train_y.shape)

# 特徴量
train_x=X.drop('meta', axis=1)
print(train_x.shape)

# 標準化
f1_columns=list(train_x.columns)

sc = StandardScaler()
dff = sc.fit_transform(train_x)
train_x=pd.DataFrame(dff)
train_x.columns=f1_columns
train_x.head()


# In[70]:


# test データ
testdf=df
print(testdf.shape)


# ラベル
test_y=np.array(testdf['meta'])
test_y=np.reshape(test_y, (len(test_y),))            # クラス分類を整数値のベクトルで表現したもの
n_labels = len(np.unique(test_y))  # 分類クラスの数 = 5
test_y=np.eye(n_labels)[test_y]
test_y=np.reshape(test_y, (len(test_y), 91))
print(test_y.shape)

# 特徴量
test_x=testdf.drop('meta', axis=1)
print(test_x.shape)


# 標準化
f1_columns=list(test_x.columns)

sc = StandardScaler()
dff = sc.fit_transform(test_x)
test_x=pd.DataFrame(dff)
test_x.columns=f1_columns
test_x.head()


# train and test data sie
print('train_size:{}, {}'.format(train_x.shape, train_y.shape))
print('train_size:{}, {}'.format(test_x.shape, test_y.shape))



# Neural Network train
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
K.set_session(sess)
classes=91
model = Sequential()
model.add(Dense(2048, input_dim=6, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(classes, activation='softmax'))
# model.load_weights('/home/ubuntu/c_point/nn_ep10.h5')
model.compile(optimizer='Adam',
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])

model.fit(train_x, train_y, epochs=30)
"""

Epoch 19/20
25935/25935 [==============================] - 33s 1ms/step - loss: 0.4993 - acc: 0.8645
Epoch 20/20
25935/25935 [==============================] - 31s 1ms/step - loss: 0.4466 - acc: 0.8857
"""


# test
_, acc =model.evaluate(test_x,test_y, verbose=1)
print('\nTest accuracy: {0}'.format(acc))

y_pred = model.predict(test_x,verbose=1)
LABEL=np.array(testdf['meta'])
top_k=sess.run(tf.nn.top_k(y_pred,k=1,sorted=False))


# evaluation

label=np.array(LABEL)
print(label.shape)

f_pred=np.argmax(y_pred, axis=1)
print(f_pred.shape)

print('acuracy:{}'.format(accuracy_score(label,f_pred)))
label_string = ['{}'.format(i) for i in range(91)]
print(classification_report(label, f_pred,target_names=label_string))


# モデル保存
model.save('/home/ubuntu/c_point/cfil_model.h5')
