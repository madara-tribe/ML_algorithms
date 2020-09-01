import cv2
import numpy as np
from torchvision import models
import torch
from torch import nn


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img =         np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
def ToNumpy(tensor):
    return tensor.to('cpu').detach().numpy().copy()

def ToTensor(numpy_):
    return torch.from_numpy(numpy_).clone()


def main():
    path = 't.jpeg'
    model = models.resnet50(pretrained=True)
    img = cv2.imread(path, 1)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    input = preprocess_image(img)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # extract feature
    model.fc = Identity() 
    inputs = input.to(device)
    output = model(inputs)
    
    # convert numpy and save
    outputs = ToNumpy(output)
    np.save('np_save', outputs)
    
if __name__ == '__main__':
    main()

