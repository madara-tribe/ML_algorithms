#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import os
import numpy as np
import torch
from tqdm import tqdm
import glob
import cv2
from FeatureExtractor import Identity, ToNumpy, ToTensor
import resnet


# In[2]:


def load_premodel(pretrain_path):
    print('loading pretrained model {}'.format(pretrain_path))
    model = resnet.generate_model(model_depth=152,
                                          n_classes=700,
                                          n_input_channels=3,
                                          shortcut_type='B',
                                          conv1_t_size=7,
                                          conv1_t_stride=1,
                                          no_max_pool=False,
                                          widen_factor=1.0)

    pretrain = torch.load(pretrain_path, map_location='cpu')
    model.load_state_dict(pretrain['state_dict'])
    #print(model)
    return model
def sample_extract(model_):
    img = torch.empty(1, 3, 16, 224, 224)
    model_.fc = Identity()
    outputs = model_(img)
    print(outputs.shape)


# In[4]:


class ModelToSave:
    def __init__(self, model, frames=16, resize_height=112, resize_width=112):
        self.model = model
        self.frames = frames
        self.resize_height = resize_height
        self.resize_width = resize_width
        
        
    def saveToNumpy(self, buffer, new_dir, file_count):
        buffer = np.array(buffer).reshape(1, 3, self.frames, self.resize_height, self.resize_width)
        inputs = ToTensor(buffer) # torch.from_numpy(bf)
        # extract feature
        tmp_model = self.model
        tmp_model.fc = Identity() 
        preds = tmp_model(inputs.float())
        
        #print('save dir', os.path.join(new_dir, 'npy_{}'.format(file_count)))
        preds = ToNumpy(preds)
        #print('preds', preds.shape, type(preds))
        np.save(os.path.join(new_dir, 'npy_{}'.format(file_count)), preds)
        print('saved')

    def save_run(self, root_path = '/home/ubuntu/data/ucf/jpg', new_root_path = '/home/ubuntu/R'):
        buffer = []
        for videos_path in tqdm(sorted(glob.glob(root_path+'/*/*'))):
            
            file_count = 0
            labels_video_path = videos_path.replace(root_path+'/', '')
            new_path = os.path.join(new_root_path, labels_video_path)
            #print(new_path)
            os.makedirs(new_path, exist_ok=True)
            for jpg_path in sorted(glob.glob(videos_path+'/*')):
                
                frame = cv2.imread(jpg_path)
                frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                buffer.append(frame)
                if len(buffer) == self.frames:
                    #print('pass', len(buffer))
                    # reset
                    self.saveToNumpy(buffer, new_path, file_count)
                    buffer = []
                    file_count += 1
            

if __name__ == '__main__':
    pretrain_path = '/home/ubuntu/results/r3d152_K_200ep.pth'
    model = load_premodel(pretrain_path)
    env = ModelToSave(model)
    env.save_run()


# In[ ]:




