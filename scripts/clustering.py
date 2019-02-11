import numpy as np
import pylab as plt
from sklearn.cluster import KMeans
import cv2
import os

prefix = '/home/ajuska/Dokumenty/Skola/diplomka/segmentation/imgs/1/'
img = cv2.imread(os.path.join(prefix, 'Study_02_00007_01_L_registered.avi_average_image.tif'))

X = np.reshape(img, (img.shape[0]*img.shape[1], img.shape[2]))

N_clus = 5
km = KMeans(n_clusters=N_clus, init='random')
km.fit(X.astype(float))
labels = np.reshape(km.labels_, img.shape[0:2])

plt.figure()
plt.imshow(img)
for l in range(N_clus):
    plt.contour(labels == l, colors=[plt.cm.nipy_spectral(l / float(N_clus))])
plt.xticks(())
plt.yticks(())
plt.show()
