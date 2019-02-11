import cv2
import os
from matplotlib import pyplot as plt

prefix = '/home/ajuska/Dokumenty/Skola/diplomka/segmentation/imgs/1/'

img = cv2.imread(os.path.join(prefix, 'Study_02_00007_01_L_registered.avi_average_image.tif'))
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_smooth = cv2.medianBlur(gray, 5)
thresh2 = cv2.adaptiveThreshold(img_smooth, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

plt.figure(figsize=(15, 15))
plt.imshow(thresh2, cmap='gray')
plt.show()

