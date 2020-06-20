import os
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.callbacks import *
from keras.layers import TimeDistributed
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam
from Residual_Prednet import PredNet
from utils.DataGenerator import SequenceGenerator
from tensorflow.python.client import device_lib
import cv2
import warnings
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings('ignore')
device_lib.list_local_devices()


def extrap_loss(y_true, y_hat):
    y_true = y_true[:, 1:]
    y_hat = y_hat[:, 1:]
    return 0.5 * K.mean(K.abs(y_true - y_hat), axis=-1)  # 0.5 to match scale of loss when trained in error mode (positive and negative errors split)

orig_weights_file="first_weight/07_prednet.hdf5"
extrap_weights_file = "first_weight/44_ExtrapFinetune.hdf5"

test_file = "test_images.npy"

# Training parameters
H=160
W=128
extrap_start_time = 96  # starting at this time step, the prediction from the previous time step will be treated as the actual input
batch_size = 1
nt = 120



X=np.load(test_file)
plt.imshow(X[1], "gray"),plt.show()
X = [cv2.resize(img, (W, H)) for img in X]
test_y = np.array(X).reshape(len(X), H, W, 1)
plt.imshow(test_y[1].reshape(H, W), "gray"),plt.show()
print(test_y.shape)
print(test_y.min(), test_y.max())
np.unique(test_y)


# prednet 

n_channels, im_height, im_width = (1, H, W)
input_shape = (im_height, im_width, n_channels)
stack_sizes = (n_channels, 48, 96, 192)
R_stack_sizes = stack_sizes
A_filt_sizes = (3, 3, 3)
Ahat_filt_sizes = (3, 3, 3, 3)
R_filt_sizes = (3, 3, 3, 3)
layer_loss_weights = np.array([1., 0., 0., 0.])  # weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
time_loss_weights = 1./ (nt - 1) * np.ones((nt,1))  # equally weight all timesteps except the first
time_loss_weights[0] = 0

prednet = PredNet(stack_sizes, R_stack_sizes,
                  A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                  output_mode='error', return_sequences=True)


inputs = Input(shape=(nt,) + input_shape)
errors = prednet(inputs)  # errors will be (batch_size, nt, nb_layers)
errors_by_time = TimeDistributed(Dense(1, trainable=False), weights=[layer_loss_weights, np.zeros(1)], trainable=False)(errors)  # calculate weighted error by layer
errors_by_time = Flatten()(errors_by_time)  # will be (batch_size, nt)
final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)  # weight errors by time
orig_model = Model(inputs=inputs, outputs=final_errors)
#orig_model.compile(loss='mean_absolute_error', optimizer='adam')
orig_model.load_weights(orig_weights_file)
orig_model.summary()


# finetune


layer_config = orig_model.layers[1].get_config()
layer_config['output_mode'] = 'prediction'
layer_config['extrap_start_time'] = extrap_start_time
data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
prednet = PredNet(weights=orig_model.layers[1].get_weights(), **layer_config)

input_shape = list(orig_model.layers[0].batch_input_shape[1:])
input_shape[0] = nt
inputs = Input(input_shape)
predictions = prednet(inputs)
extrap_model = Model(inputs=inputs, outputs=predictions)
#extrap_model.compile(loss=extrap_loss, optimizer=Adam(lr=0.0001))
extrap_model.load_weights(extrap_weights_file)
extrap_model.summary()



test_gene = SequenceGenerator(test_y, H, W, nt, batch_size, output_mode='prediction', sequence_start_mode='unique')
X_test=test_gene.create_all()
print(X_test.shape)


# predict
X_hat = extrap_model.predict(X_test, batch_size)
print(X_test.shape, X_hat.shape)


# accuracy
mse_model = np.mean( (X_test[:, 1:] - X_hat[:, 1:])**2 )  # look at all timesteps except the first
mse_prev = np.mean( (X_test[:, :-1] - X_test[:, 1:])**2 )
print("mse_model", mse_model ,"mse_prev", mse_prev)
RESULTS_SAVE_DIR="extrap"

# reshape for plot
X_tests = X_test.reshape(len(X_test), nt, H, W)
X_hats = X_hat.reshape(len(X_hat), nt, H, W)
plt.imshow(X_tests[1][1], "gray"),plt.show()
plt.imshow(X_hats[1][1], "gray"),plt.show()
print(X_tests.shape, X_hats.shape) 

# mse_model 0.013940161 mse_prev 0.012056994
# mse_model 0.04286805 mse_prev 0.0412904



# Plot some predictions and save
n_plot = 40
aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
plt.figure(figsize = (nt, 2*aspect_ratio))
gs = gridspec.GridSpec(2, nt)
gs.update(wspace=0., hspace=0.)
plot_save_dir = os.path.join(RESULTS_SAVE_DIR, 'prediction_plots/')
if not os.path.exists(plot_save_dir): os.mkdir(plot_save_dir)
plot_idx = np.random.permutation(X_tests.shape[0])[:n_plot]

for i in plot_idx:
    for t in range(nt):
        plt.subplot(gs[t])
        plt.imshow(X_tests[i,t], interpolation='none')
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t==0: plt.ylabel('Actual', fontsize=10)

        plt.subplot(gs[t + nt])
        plt.imshow(X_hats[i,t], interpolation='none')
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t==0: plt.ylabel('Predicted', fontsize=10)

    plt.savefig(plot_save_dir +  'plot_' + str(i) + '.png')
    plt.clf()


# check plot
c=0
for test, pred in zip(X_tests[0], X_hats[0]):
    print(c, "test")
    print(test.shape)
    plt.imshow(test, "gray"),plt.show()
    print(c, "pred")
    plt.imshow(pred, "gray"),plt.show()
    c+=1


# evaluation by images as eval criteria


pred1="weather_criteria/pred_span2.npy"

Y=np.load(pred1)
#Y[Y<90]=0
plt.imshow(Y[1], "gray"),plt.show()
Y = [cv2.resize(img, (W, H)) for img in Y]
evals = np.array(Y).reshape(len(Y), H, W, 1)
plt.imshow(evals[1].reshape(H, W), "gray"),plt.show()
print(evals.shape)
print(evals.min(), evals.max(), np.unique(evals))


# eval2


pred_span=nt-extrap_start_time

X_all = evals.reshape(1, 24, H, W, 1)
zeros = np.zeros((1, extrap_start_time-24) + (H, W, 1), np.float32)
azeros = np.zeros((1, 24) + (H, W, 1), np.float32)
print(zeros.shape, X_all.shape, azeros.shape)
TX=np.hstack([zeros, X_all, azeros])
TX=TX/255
print(TX.shape)


# predict
TX_hat = extrap_model.predict(TX, batch_size)
print(TX.shape, TX_hat.shape)


# check to plot


for c, (pred, test) in enumerate(zip(TX_hat[0][90:], TX[0][90:])):
    print('test', c)
    plt.imshow(test.reshape(H, W), 'gray'),plt.show()
    print('pred', c)
    plt.imshow(pred.reshape(H, W), 'gray'),plt.show()
