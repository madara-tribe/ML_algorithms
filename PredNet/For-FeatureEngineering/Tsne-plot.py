import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import cv2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
import glob

"""2d plot"""
def plot2d(IMG):
    # conver 2-d  # len() =>4781
    images = np.concatenate([img.flatten().reshape(1,-1) for img in IMG], axis=0)

    # TSNE
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(images)


    # label
    k=np.array([0]*len(images))
    nlabel=np.reshape(k, (len(images),))
    print('normal_label:{}'.format(nlabel.shape), nlabel[:1])


    # plot
    def number_plot(X, Y, shade=None):
        markers = ["$0$", "$1$", "$2$", "$3$", "$4$", "$5$", "$6$", "$7$", "$8$", "$9$"]
        plt.figure(figsize=(10,10))
        if shade is not None:
            plt.scatter(shade[:, 0], shade[:, 1])
        for i in range(10):
            X_paint = X[Y == i]
            plt.scatter(X_paint[:, 0], X_paint[:, 1], marker=markers[i])
        plt.show()

    number_plot(X_tsne, nlabel)

path='/home/ubuntu/test_image/*/*/*/*.png'
IMG=[]
for files in glob.glob(path):
    img = cv2.imread(files, 0)
    img1 = img[40:40+420, 130:130+340]
    IMG.append(img1)

plot2d(IMG)


"""3d TSNE plot"""．
def preprocess_image(path):
    img = cv2.imread(files, 0)
    resized = img[40:40+420, 130:130+340]
    normalized = cv2.normalize(resized, None, 0.0, 1.0, cv2.NORM_MINMAX)
    timg = normalized.reshape(np.prod(normalized.shape))
    return timg/np.linalg.norm(timg)



preprocess_images_as_vecs = [preprocess_image(p) for p in path]
# tsne
tsne = TSNE(
    n_components=3, #ここが削減後の次元数です．
    init='random', random_state=101, method='barnes_hut',
    n_iter=1000, verbose=2).fit_transform(preprocess_images_as_vecs)


# 3Dの散布図が作れるScatter3d
trace1 = go.Scatter3d(
    x=tsne[:,0], # それぞれの次元をx, y, zにセット．
    y=tsne[:,1],
    z=tsne[:,2],
    mode='markers',
    marker=dict(
        sizemode='diameter',
        color = preprocessing.LabelEncoder().fit_transform(nlabel),
        colorscale = 'Portland',
        line=dict(color='rgb(255, 255, 255)'),
        opacity=0.9,
        size=2
    )
)

data=[trace1]
layout=dict(height=700, width=600, title='coil-20 tsne exmaple')
fig=dict(data=data, layout=layout)
offline.iplot(fig, filename='tsne_example')
