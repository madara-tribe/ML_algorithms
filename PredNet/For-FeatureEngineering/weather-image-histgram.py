import os
conda_file_dir = "/Users/anaconda3/envs/dum_env"
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib
from mpl_toolkits.basemap import Basemap
import numpy as np
import cv2
import pandas as pd
import os
import gzip
import shutil
import mpl_toolkits
from datetime import datetime as dt
from datetime import timedelta
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
import glob


"""month  png image histgram"""
main_path="/Users/downloads/competition"
month1='4'
month2='5'
month3='6'
img01 = []
img02 = []
img03 = []
# 1ヶ月分の衛星画像を読み込む
path1='train-sat_image_2016_01/*/2016-04-*/*.png'
for file in glob.glob(os.path.join(main_path, path1)):
    if os.path.exists(file):
        img = cv2.imread(file, 0)
        img01.append(img)

path2='train-sat_image_2016_01/*/2016-05-*/*.png'
for file2 in glob.glob(os.path.join(main_path, path2)):
    if os.path.exists(file2):
        img = cv2.imread(file2, 0)
        img02.append(img)

path3='train-sat_image_2016_01/*/2016-06-*/*.png'
for file3 in glob.glob(os.path.join(main_path, path3)):
    if os.path.exists(file3):
        img = cv2.imread(file3, 0)
        img03.append(img)

# ヒストグラム作成
all_img01 = np.concatenate(img01).flatten()
all_img02 = np.concatenate(img02).flatten()
all_img03 = np.concatenate(img03).flatten()

fig, ax = plt.subplots(1, 1, figsize=(8,8))
ax.hist(all_img01, bins=np.arange(256 + 1), alpha=0.3, color="r", label=month1+", 2016")
ax.hist(all_img02, bins=np.arange(256 + 1), alpha=0.3, color="b", label=month2+", 2016")
ax.hist(all_img03, bins=np.arange(256 + 1), alpha=0.3, color="g", label=month3+", 2016")
ax.set_title("Histogram of Satellite Images")
ax.set_ylim(0, 6000000)
plt.legend()
plt.show()
