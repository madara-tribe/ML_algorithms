import glob
import cv2
import os
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from FeatureExtractor import Identity, ToNumpy, ToTensor
import resnet



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
    print('model pass')
    model.load_state_dict(pretrain['state_dict'])
    #print(model)
    return model
def sample_extract(model_):
    img = torch.empty(1, 3, 16, 224, 224)
    model_.fc = Identity()
    outputs = model_(img)
    print(outputs.shape)


class ModelToSave:
    def __init__(self, models, device, frames=16, resize_height=112, resize_width=112):
        self.model = models
        self.device = device
        self.frames = frames
        self.resize_height = resize_height
        self.resize_width = resize_width
        
    def saveToNumpy(self, buffer, new_dir, file_count):        
        buffer = np.array(buffer).reshape(1, 3, self.frames, self.resize_height, self.resize_width)
        inputs = ToTensor(buffer)
        inputs = inputs.to(self.device)
        tmp_model = self.model
        tmp_model = tmp_model.to(self.device)
        tmp_model.fc = Identity() 
        preds = tmp_model(inputs.float())

        preds = ToNumpy(preds)
        print(preds.shape)
        np.save(os.path.join(new_dir, 'npy_{}'.format(file_count)), preds)
        print('saved')

    def save_run(self, root_path = 'activityNet_test', new_root_path = 'test_jpg'):
        buffer = []
        for videos_path in tqdm(sorted(glob.glob(root_path+'/*'))):
            file_count = 0
            labels_video_path = videos_path.replace(root_path+'/', '')
            new_path = os.path.join(new_root_path, labels_video_path)
            os.makedirs(new_path, exist_ok=True)
            for jpg_path in sorted(glob.glob(videos_path+'/*')):
                frame = cv2.imread(jpg_path)
                #print('f shape', frame.shape)
                if frame is None:
                    continue
                frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                buffer.append(frame)
                if len(buffer) == self.frames:
                    #print('pass', len(buffer))
                    # reset
                    self.saveToNumpy(buffer, new_path, file_count)
                    buffer = []
                    file_count += 1
                    
if __name__ == '__main__':
    GPU_ID = '1'
    device = torch.device("cuda:{}".format(GPU_ID) if torch.cuda.is_available() else "cpu")
    pretrain_path = '/home/app/space/results/r3d152_K_200ep.pth'
    model = load_premodel(pretrain_path)
    env = ModelToSave(models=model, device=device)
    env.save_run()