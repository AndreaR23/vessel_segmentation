import os
import cv2
from tqdm import tqdm
import numpy as np


class ImageUtils:
    def __init__(self, path_to_imgs_dir, path_to_masks_dir, img_height, img_width, architecture=None):
        self._path_to_imgs_dir = path_to_imgs_dir
        self._path_to_masks_dir = path_to_masks_dir
        self._img_height = img_height
        self._img_width = img_width
        self._loaded_imgs = {}
        self._loaded_masks = {}
        self._preprocessed_imgs_masks = {}
        self._architecture = architecture

    def _load_images(self):
        print('Loading images.')
        for file in tqdm(sorted(os.listdir(self._path_to_imgs_dir))):
            img = cv2.imread(os.path.join(self._path_to_imgs_dir, file))[:, :, :3]
            self._loaded_imgs[file] = img

    def _load_masks(self):
        print('\nLoading masks.')
        for key, val in self._loaded_imgs.items():
            path = os.path.join(self._path_to_masks_dir, key)
            try:
                mask = cv2.imread(os.path.join(self._path_to_masks_dir, key))[:, :, :1]
                self._loaded_masks[key] = mask
            except:
                print(key)
    
    def _normalized(self, rgb):
        #return rgb/255.0
        norm = np.zeros((rgb.shape[0], rgb.shape[1], 3), np.float32)

        b = rgb[:,:,0]
        g = rgb[:,:,1]
        r = rgb[:,:,2]

        norm[:,:,0] = cv2.equalizeHist(b)
        norm[:,:,1] = cv2.equalizeHist(g)
        norm[:,:,2] = cv2.equalizeHist(r)

        return norm
    
    def _one_hot_it(self, labels):
        x = np.zeros([self._img_height, self._img_width, 2])
        for i in range(self._img_height):
            for j in range(self._img_width):
                x[i,j,labels[i][j]] = 1
        return x
        
    def _preprocess_images(self):
        print('\nPreprocessing images.')
        for key, val in tqdm(self._loaded_imgs.items()):
#             img_ = cv2.cvtColor(val[50:50+670, 50:50+900], cv2.COLOR_BGR2GRAY)
#             img = cv2.cvtColor(cv2.equalizeHist(img_), cv2.COLOR_GRAY2RGB)
#             converted_img = cv2.cvtColor(val, cv2.COLOR_RGB2LAB)
#             converted_img[:,:,0] = cv2.equalizeHist(converted_img[:,:,0])            
            img = self._normalized(val[50:50+670, 50:50+900])
#             img = converted_img[50:50+670, 50:50+900]
            mask_ = self._loaded_masks[key]
            if self._architecture == 'unet':
                mask = mask_[50:50+670, 50:50+900]
            elif self._architecture == 'segnet':
                mask = self.one_hot_it(mask_[50:50+670, 50:50+900])
            self._preprocessed_imgs_masks[key] = (img, mask)

    def get_preprocessed_images(self):
        self._load_images()
        self._load_masks()
        self._preprocess_images()
        return self._preprocessed_imgs_masks
