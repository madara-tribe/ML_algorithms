import os
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import TimeDistributed
from keras.optimizers import Adam
from Residual_Prednet import PredNet
from utils.DataGenerator import SequenceGenerator
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.python.client import device_lib
import cv2
import warnings
warnings.filterwarnings('ignore')
device_lib.list_local_devices()


# Data files
train_file = 'train_image.npy'
test_file= "test_image.npy"
weights_file="09_prednet.hdf5"
# Training parameters
batch_size = 1
nt = 120  # number of timesteps used for sequences in training


# In[4]:


X=np.load(test_file) 
plt.imshow(X[1], "gray"),plt.show()
X = [cv2.resize(img, (128, 160)) for img in X]
test_y = np.array(X).reshape(len(X), 160, 128, 1)
plt.imshow(test_y[1].reshape(160, 128), "gray"),plt.show()
print(test_y.shape)
print(test_y.min(), test_y.max())


# In[5]:


n_channels, im_height, im_width = (1, 160, 128)
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


# In[6]:


inputs = Input(shape=(nt,) + input_shape)
errors = prednet(inputs)  # errors will be (batch_size, nt, nb_layers)
errors_by_time = TimeDistributed(Dense(1, trainable=False), weights=[layer_loss_weights, np.zeros(1)], trainable=False)(errors)  # calculate weighted error by layer
errors_by_time = Flatten()(errors_by_time)  # will be (batch_size, nt)
final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)  # weight errors by time
model = Model(inputs=inputs, outputs=final_errors)
model.load_weights(weights_file)
model.summary()


layer_config = model.layers[1].get_config()
layer_config['output_mode'] = 'prediction'
test_prednet = PredNet(weights = model.layers[1].get_weights(), **layer_config)
input_shape = list(model.layers[0].batch_input_shape[1:])
input_shape[0] = nt
inputs = Input(shape=tuple(input_shape))
predictions = test_prednet(inputs)

test_model = Model(inputs=inputs, outputs=predictions)
test_model.summary()



test_gene = SequenceGenerator(test_y, H, W, nt, batch_size, output_mode='prediction', sequence_start_mode='unique')
X_test=test_gene.create_all()
print(X_test.shape)


# predict
X_hat = test_model.predict(X_test, batch_size)
print(X_test.shape, X_hat.shape)


# accuracy
mse_model = np.mean( (X_test[:, 1:] - X_hat[:, 1:])**2 )  # look at all timesteps except the first
mse_prev = np.mean( (X_test[:, :-1] - X_test[:, 1:])**2 )
print("mse_model", mse_model ,"mse_prev", mse_prev)
RESULTS_SAVE_DIR="tb"

# reshape for plot
X_tests = X_test.reshape(len(X_test), nt, 160, 128)
X_hats = X_hat.reshape(len(X_hat), nt, 160, 128)
plt.imshow(X_tests[1][0], "gray"),plt.show()
plt.imshow(X_hats[1][0], "gray"),plt.show()
print(X_tests.shape, X_hats.shape) 
# train: mse_model 0.005529687 mse_prev 0.009819959
# test: mse_model 0.007216311 mse_prev 0.012056988
# mse_model 0.6809688 mse_prev 0.012056977



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


# check image by plot
c=0
for test, pred in zip(X_test[0], X_hat[0]):
    print(c, "test", 'shape', test.shape)
    plt.imshow(test.reshape(160, 128)*255, "gray"),plt.show()
    print(c, "pred")
    plt.imshow(pred.reshape(160, 128)*255, "gray"),plt.show()
    c+=1
