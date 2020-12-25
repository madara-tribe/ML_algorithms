import gc
from copy import deepcopy
import cv2
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime

from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, UpSampling2D, Dropout, LeakyReLU, BatchNormalization, Activation
from keras.layers.merge import Concatenate
from keras.applications import VGG16
from keras import backend as K
from libs.pconv_layer import PConv2D
from libs.util import random_mask
from losses.total_loss import PConv_totl_loss
from libs.pconv_model import PConvUnet
BATCH_SIZE = 2


def build_pconv_unet(img_rows, img_cols, train_bn=True, lr=0.0002):      
    assert img_rows >= 256, 'Height must be >=256 pixels'
    assert img_cols >= 256, 'Width must be >=256 pixels'
    inputs_img = Input((img_rows, img_cols, 3))
    inputs_mask = Input((img_rows, img_cols, 3))

    # ENCODER
    def encoder_layer(img_in, mask_in, filters, kernel_size, bn=True):
        conv, mask = PConv2D(filters, kernel_size, strides=2, padding='same')([img_in, mask_in])
        if bn:
            conv = BatchNormalization(name='EncBN'+str(encoder_layer.counter))(conv, training=train_bn)
        conv = Activation('relu')(conv)
        encoder_layer.counter += 1
        return conv, mask
    encoder_layer.counter = 0

    e_conv1, e_mask1 = encoder_layer(inputs_img, inputs_mask, 64, 7, bn=False)
    e_conv2, e_mask2 = encoder_layer(e_conv1, e_mask1, 128, 5)
    e_conv3, e_mask3 = encoder_layer(e_conv2, e_mask2, 256, 5)
    e_conv4, e_mask4 = encoder_layer(e_conv3, e_mask3, 256, 3)
    e_conv5, e_mask5 = encoder_layer(e_conv4, e_mask4, 256, 3)
    e_conv6, e_mask6 = encoder_layer(e_conv5, e_mask5, 256, 3)
    e_conv7, e_mask7 = encoder_layer(e_conv6, e_mask6, 256, 3)
    e_conv8, e_mask8 = encoder_layer(e_conv7, e_mask7, 256, 3)

    # DECODER
    def decoder_layer(img_in, mask_in, e_conv, e_mask, filters, kernel_size, bn=True):
        up_img = UpSampling2D(size=(2,2))(img_in)
        up_mask = UpSampling2D(size=(2,2))(mask_in)
        concat_img = Concatenate(axis=3)([e_conv,up_img])
        concat_mask = Concatenate(axis=3)([e_mask,up_mask])
        conv, mask = PConv2D(filters, kernel_size, padding='same')([concat_img, concat_mask])
        if bn:
            conv = BatchNormalization()(conv)
        conv = LeakyReLU(alpha=0.2)(conv)
        return conv, mask

    d_conv9, d_mask9 = decoder_layer(e_conv8, e_mask8, e_conv7, e_mask7, 256, 3)
    d_conv10, d_mask10 = decoder_layer(d_conv9, d_mask9, e_conv6, e_mask6, 256, 3)
    d_conv11, d_mask11 = decoder_layer(d_conv10, d_mask10, e_conv5, e_mask5, 256, 3)
    d_conv12, d_mask12 = decoder_layer(d_conv11, d_mask11, e_conv4, e_mask4, 256, 3)
    d_conv13, d_mask13 = decoder_layer(d_conv12, d_mask12, e_conv3, e_mask3, 256, 3)
    d_conv14, d_mask14 = decoder_layer(d_conv13, d_mask13, e_conv2, e_mask2, 128, 3)
    d_conv15, d_mask15 = decoder_layer(d_conv14, d_mask14, e_conv1, e_mask1, 64, 3)
    d_conv16, d_mask16 = decoder_layer(d_conv15, d_mask15, inputs_img, inputs_mask, 3, 3, bn=False)
    outputs = Conv2D(3, 1, activation = 'sigmoid')(d_conv16)        

    # Setup the model inputs / outputs
    model = Model(inputs=[inputs_img, inputs_mask], outputs=outputs)

    # Compile the model
    model.compile(
        optimizer = Adam(lr=lr),
        loss="mse"#self.loss_total(inputs_mask)
    )

    return model

#pmodel = build_pconv_unet(256, 256)
#pmodel.summary()


path= '/Users/hagi/downloads/images_1024_pickled_r3_fmap_1024'
X=np.load(path+'/test_PCs_x.npy')
print(X.shape, X.max())
plt.imshow(X[1]),plt.show()


# In[20]:


class DataGenerator(ImageDataGenerator):
    def flow(self, x, *args, **kwargs):
        while True:
            
            # Get augmentend image samples
            ori = next(super().flow(x, *args, **kwargs))

            # Get masks for each image sample
            mask = np.stack([random_mask(ori.shape[1], ori.shape[2]) for _ in range(ori.shape[0])], axis=0)

            # Apply masks to all image sample
            masked = deepcopy(ori)
            masked[mask==0] = 1

            # Yield ([ori, masl],  ori) training batches
            # print(masked.shape, ori.shape)
            gc.collect()
            yield [masked, mask], ori        

# Create datagen
train_datagen = DataGenerator(rotation_range=10,
                              #width_shift_range=0.2,
                              #height_shift_range=0.2,
                              zoom_range = [0.9,1],
                              horizontal_flip=True
                              )

# Create generator from numpy arrays
train_generator = train_datagen.flow(x=X, batch_size=BATCH_SIZE)

# Create datagen
test_datagen = DataGenerator(horizontal_flip=True)

# Get samples & Display them
test_generator = test_datagen.flow(x=X, batch_size=BATCH_SIZE)
(masked, mask), ori = next(test_generator)
plt.imshow(masked[0]),plt.show()
plt.imshow(mask[0]),plt.show()
plt.imshow(ori[0]),plt.show()
def plot_callback(model):
    """Called at the end of each epoch, displaying our previous test images,
    as well as their masked predictions and saving them to disk"""
    
    # Get samples & Display them        
    pred_img = model.predict([masked, mask])

    # Clear current output and display test images
    for i in range(len(ori)):
        _, axes = plt.subplots(1, 3, figsize=(20, 5))
        axes[0].imshow(masked[i,:,:,:])
        axes[1].imshow(pred_img[i,:,:,:] * 1.)
        axes[2].imshow(ori[i,:,:,:])
        axes[0].set_title('Masked Image')
        axes[1].set_title('Predicted Image')
        axes[2].set_title('Original Image')                
        plt.show()




## setting total loss
total_loss = PConv_totl_loss()
p_model = PConvUnet(img_rows=256, img_cols=256, total_loss=False, weight_filepath='/Users/hagi/desktop/W/')

#p_model.load('/Users/hagi/desktop/W/1_weights_2020-09-24-10-42-50.h5')
p_model.fit(train_generator,
            steps_per_epoch=10,
            epochs=5,
            plot_callback=plot_callback)




