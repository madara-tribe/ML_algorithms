import os
import numpy as np
from datetime import datetime as dt
from datetime import timedelta
import cv2
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob


class get_satelliter_TainDataset:
    def __init__(self, grayscale=True):
        self.grayscale = grayscale
        
    def load_image(self, year, main_path):
        X=[]
        fpath = main_path
        year = str(year)
        months = [str(m) if len(str(m))==2 else "0"+str(m) for m in range(1, 13)]
        days = [str(d) if len(str(d))==2 else "0"+str(d) for d in range(1, 32)]
        hours = [str(h) if len(str(h))==2 else "0"+str(h) for h in range(24)]
        for month in tqdm(months):
            for day in days:
                for hour in hours:
                    spath =os.path.join(fpath, year+"-"+month+"-"+day)
                    sdata = year+"-"+month+"-"+day
                    tpath="*"+month+"-"+day+"-"+hour+"-*.png"
                    for files in glob.glob(os.path.join(spath, tpath)):
                        if self.grayscale:
                            img = cv2.imread(files, 0)
                        else:
                            img = cv2.imread(files)
                        if img is not None:
                            img1 = img[40:40+420, 130:130+340]
                            X.append(img1)
                            print(files, img1.shape)
        return X 
    def load_all_year(self):
        X2016_1 = self.load_image(2016, "/home/ubuntu/sat/train_img/sat_image_2016_01/*")
        X2016_2 = self.load_image(2016, "/home/ubuntu/sat/train_img/sat_image_2016_02/*")
        X2017_1 = self.load_image(2017, "/home/ubuntu/sat/train_img/sat_image_2017_01/*")
        X2017_2 = self.load_image(2017, "/home/ubuntu/sat/train_img/sat_image_2017_02/*")
        train=np.vstack([X2016_1, X2016_2, X2017_1, X2017_2])
        print(train.shape)
        return (train).astype(np.float32)


load_images=get_satelliter_TainDataset(True)
X = load_images.load_all_year()
np.save('train_image', X)


class get_satelliter_TestDataset:
    def __init__(self, main_path, grayscale=True):
        self.grayscale = grayscale
        self.main_path = main_path
    def load_image(self):
        X=[]
        year = str(2018)
        fpath=self.main_path
        months = [str(m) if len(str(m))==2 else "0"+str(m) for m in range(1, 13)]
        days = [str(d) if len(str(d))==2 else "0"+str(d) for d in range(1, 32)]
        hours = [str(h) if len(str(h))==2 else "0"+str(h) for h in range(24)]
        for month in tqdm(months):
            for day in days:
                for hour in hours:
                    spath =os.path.join(fpath, year+"-"+month+"-"+day)
                    tpath="*"+month+"-"+day+"-"+hour+"-*.png"
                    for files in glob.glob(os.path.join(spath, tpath)):
                        if self.grayscale:
                            img = cv2.imread(files, 0)
                        else:
                            img = cv2.imread(files)
                        if img is not None:
                            img1 = img[40:40+420, 130:130+340]
                            X.append(img1)
                            print(files, img1.shape)
        Xs = np.array(X, dtype=np.float32)
        print(Xs.shape)
        return Xs

main_path = "/home/ubuntu/sat/test_img/sat_image_2018/*/"
test_load=get_satelliter_TestDataset(main_path, True)
y = test_load.load_image()
print(y.shape)


np.save('test_image', y)
