# conda install basemap
# conda info >> active env location : /anaconda3/envs/rot
import os
conda_file_dir = "/home/ubuntu/anaconda3/envs/tensorflow_p36"
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib
# from mpl_toolkits.basemap import Basemap

import numpy as np
import cv2
import pandas as pd
import glob
import os
import gzip
import shutil
import mpl_toolkits
from datetime import datetime as dt
from datetime import timedelta
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
import warnings


def Read_gz_Binary(file):
    file_tmp = file + "_tmp"
    with gzip.open(file, 'rb') as f_in:
        with open(file_tmp, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    bin_data = np.fromfile(file_tmp, np.float32)
    os.remove(file_tmp)
    return bin_data.reshape( [168,128] )



def fill_lack_data(data):
    # まず欠損行の穴埋めは、値が存在する上下端の行の値をそのままコピーする
    data[0:2] = data[2]
    data[154:] = data[153]

    # 欠損列の穴埋めも、値が存在する左右端の列の値をそのままコピーする
    data[:, :8] = data[:, 8].reshape(-1,1)
    return data




# Ns month
X=[]
y=[]
cs=[3,3,0,0,0,1]
main_path="/home/ubuntu/meta/test_meta/test-met_data_2018_01"
for c, i in zip(cs, range(1, 7)):
    if i<10:
        path='met/*/0'+str(i)+'/*/*.gz'
    else:
        path='met/*/'+str(i)+'/*/*.gz'
    print(path, c)
    for files in glob.glob(os.path.join(main_path, path)):
        if 'TMP' in files:
            lacked_file = Read_gz_Binary(files)
            file = fill_lack_data(lacked_file)
            X.append(file-273.15)
            y.append(c)
            # draw_weather_map(file-273.15, draw="shaded", levels=list(np.arange(-50, 50, 3)))
        if 'PRMSL.msl' in files:
            psmsl_file = Read_gz_Binary(files)
            psmsl = fill_lack_data(psmsl_file)
            X.append(psmsl/100)
            y.append(c)
            # draw_weather_map(psmsl/100, draw="contour", levels=list(np.arange(992, 1028, 4)))
        if '/RH.' in files:
            rh_file = Read_gz_Binary(files)
            rh = fill_lack_data(rh_file)
            X.append(rh)
            y.append(c)
            # draw_weather_map(rh, draw="shaded", levels=list(np.arange(0, 101, 5)))
        if '/HGT' in files:
            hgt_file = Read_gz_Binary(files)
            hgt = fill_lack_data(hgt_file)
            X.append(hgt)
            y.append(c)
            # draw_weather_map(hgt, draw="contour", levels=list(np.arange(1000, 10000, 60)))
        if "UGRD" in files:
            ugrd_file = files
            lack_ugrd = Read_gz_Binary(ugrd_file)
            ugrd = fill_lack_data(lack_ugrd)
            X.append(ugrd/0.5144)
            y.append(c)
        if "VGRD" in files:
            vgrd_file = files
            lack_vgrd = Read_gz_Binary(vgrd_file)
            vgrd = fill_lack_data(lack_vgrd)
            X.append(vgrd/0.5144)
            y.append(c)
            # draw_weather_map(hgtw, u=ugrd/0.5144, v=vgrd/0.5144, draw="barb", levels=list(np.arange(8400,9840,120)))





X=np.array(X)
y=np.array(y)
print(X.shape, y.shape)

# save
np.save("test_x", X)
np.save("test_y", y)
