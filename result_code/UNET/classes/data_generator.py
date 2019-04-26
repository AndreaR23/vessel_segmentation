import sys
sys.path.append('..')

from imgaug import augmenters as iaa
import os
import numpy as np
import random
import math
from segmentation_models.backbones import get_preprocessing
from classes.image_utils import ImageUtils
from classes.create_patches import PatchesCreator


class DataGenerator:
    
    def __init__(self, path_to_imgs, path_to_masks, img_height, img_width, backbone, batch_size, preprocess=True,  
                 patches=True):
        self._path_to_imgs = path_to_imgs
        self._path_to_masks = path_to_masks
        self._img_height = img_height
        self._img_width = img_width
        self._backbone = backbone
        self._batch_size = batch_size
        self._preprocess = preprocess
        self._region = seg_region
        self._patches = patches

    def data_generate(self):
        image_ut = ImageUtils(self._path_to_imgs, self._path_to_masks, self._img_height, self._img_width)
        preprocess_input = get_preprocessing(self._backbone)
        
        while 1:
            img = []
            mask = []       
            names = os.listdir(self._path_to_imgs)
            random.shuffle(names)
            
            for idx in range(6):
                train_img, train_mask = image_ut.load_image_mask_pair(names[idx])
                resized_img, resized_mask = image_ut.preprocess_image_mask_pair(train_img, train_mask)
                img.append(resized_img)
                mask.append(resized_mask)           

            augmented_imgs, augmented_masks = image_ut.augment_image_mask_pair(img, mask, how_many=3)
         
            for idx in range(len(augmented_imgs)):
                img.append(augmented_imgs[idx])
                mask.append(augmented_masks[idx])
                
            if self._patches:
                all_patches_im = []
                all_patches_ma = []
                for idx in range(len(img)):
                    creator = PatchesCreator(self._img_height, self._img_width)
                    patches_im, patches_ma = creator.create_patches(img[idx], mask[idx])
                    for p_idx in range(len(patches_im)):
                        all_patches_im.append(patches_im[p_idx])
                        all_patches_ma.append(patches_ma[p_idx])
                img = all_patches_im
                mask = all_patches_ma
                    
            if self._preprocess:
                imgs = [preprocess_input(x)/255. for x in img]
                masks = [y/255. for y in mask]
            else:
                imgs = [x/255. for x in img]
                masks = [y/255. for y in mask]
            
            imgs_shuffled, masks_shuffled = image_ut.shuffle_image_mask_pairs(imgs, masks)

            cnt_im = math.floor(len(imgs_shuffled)/ self._batch_size)
            inkr = 0
            for i in range(cnt_im):
                start = 0 + inkr
                stop = 0 + inkr + self._batch_size
                batch_imgs = imgs_shuffled[start:stop]
                batch_masks = masks_shuffled[start:stop]
                inkr = inkr +  self._batch_size
                yield np.array(batch_imgs), np.array(batch_masks)
    
 