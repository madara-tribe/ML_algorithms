import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib
import matplotlib.pyplot as plt


class SequenceGenerator(object):
    def __init__(self, image, H, W, nt, batch_size, output_mode="error", sequence_start_mode='all'):
        self.image = image
        self.H = H
        self.W = W
        self.nt = nt
        self.batch_size= batch_size
        self.index_array = None
        assert output_mode in {'error', 'prediction'}, 'output_mode must be {error or prediction}'
        self.output_mode = output_mode
        assert sequence_start_mode in {'all', 'unique'}, 'sequence_start_mode must be {all or unique}'
        self.sequence_start_mode = sequence_start_mode
        if self.sequence_start_mode=='all':
            self.possible_starts = np.array([i for i in range(self.image.shape[0]-self.nt) if i%self.nt==0])
        if self.sequence_start_mode=="unique":
            possible_start = np.array([i for i in range(self.image.shape[0]-self.nt) if i%self.nt==0])
            # print(possible_start)
            self.possible_starts = np.random.permutation(possible_start)
        self.N_seq = len(self.possible_starts)

    def preprocess(self, X):
        return X.astype(np.float32)/255

    def create_all(self):
        X_all = np.zeros((self.N_seq, self.nt) + (self.H, self.W, 1), np.float32)
        for i, idx in enumerate(self.possible_starts):
            X_all[i] = self.preprocess(self.image[idx:idx+self.nt])
        return X_all

    def flow_from_img(self):
        batch_x = np.zeros((self.batch_size, self.nt) + (self.H, self.W, 1), np.float32)
        while True:
            self.index_array = np.array([self.possible_starts[i*self.batch_size:(i+1)*self.batch_size] for i in range(len(self.possible_starts))])
            for total, idxs in enumerate(self.index_array):
                for i, idx in enumerate(idxs):
                    batch_x[i] = self.preprocess(self.image[idx:idx+self.nt])
                if len(batch_x)==self.batch_size:
                    input_img=batch_x
                    # plt.imshow(input_img[1][1].reshape(128, 160), "gray"),plt.show()
                    if self.output_mode == 'error':
                        input_label = np.zeros(self.batch_size, np.float32)
                    elif self.output_mode == 'prediction':
                        input_label = input_img
                    yield input_img, input_label
