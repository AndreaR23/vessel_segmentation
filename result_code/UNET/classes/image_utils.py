import os
import cv2
from tqdm import tqdm
import numpy as np
import random
from imgaug import augmenters as iaa


class ImageUtils:
    def __init__(self, path_to_imgs_dir, path_to_masks_dir, img_height, img_width, architecture=None):
        self._path_to_imgs_dir = path_to_imgs_dir
        self._path_to_masks_dir = path_to_masks_dir
        self._img_height = img_height
        self._img_width = img_width

    def load_image_mask_pair(self, img_name):
        img = cv2.imread(os.path.join(self._path_to_imgs_dir, img_name))
        mask = cv2.imread(os.path.join(self._path_to_masks_dir, img_name), cv2.IMREAD_GRAYSCALE)
        return img, mask

    def preprocess_image_mask_pair(self, img, mask):
        resized_img =  cv2.resize(img, (self._img_height, self._img_width))

        resized_mask = cv2.resize(mask, (self._img_height, self._img_width))
        resized_mask[resized_mask > 50] = 255
        resized_mask[resized_mask <= 50] = 0
        resized_mask = resized_mask.reshape(self._img_height, self._img_width, 1)
        return resized_img, resized_mask

    def preprocess_image(self, img):
        resized_img =  cv2.resize(img, (self._img_height, self._img_width))
        return resized_img, resized_mask

    def normalized(self, rgb_img):
        #return rgb/255.0
        image_lab = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(image_lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        merged_channels = cv2.merge((cl, a_channel, b_channel))
        final_image = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2BGR)
        return final_image

    def _augmentation(self):
        # augmenting set of images and masks
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        seq = iaa.Sequential([iaa.Fliplr(0.5),
                              iaa.Crop(percent=(0, 0.1)), 
                              iaa.GaussianBlur((0, 1.0)), 
                              iaa.ContrastNormalization((0.9, 1.1)), 
                              iaa.Multiply((0.9, 1.1), per_channel=0.1), 
                              sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1))), 
                              iaa.Affine(scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                                         translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                                         rotate=(-0.01, 0.01),
                                         shear=(-8, 8))], 
                             random_order=True) 
        return seq   
    
    def augment_image_mask_pair(self, imgs_list, masks_list, how_many=3):
        
        all_aug_im = []
        all_aug_ma = []
        for i in range(how_many):
            seq = self._augmentation()
            seq_det = seq.to_deterministic()
            images_aug = seq_det.augment_images(np.array(imgs_list).astype(np.uint8))    
            mask_aug = seq_det.augment_images(np.array(masks_list).astype(np.uint8))
            mask_aug[mask_aug > 50] = 255
            mask_aug[mask_aug <= 50] = 0
            for im in range(len(images_aug)):
                all_aug_im.append(images_aug[im])
                all_aug_ma.append(mask_aug[im])
        return all_aug_im, all_aug_ma

    def shuffle_image_mask_pairs(self, imgs_list, masks_list):
        imgs_shuffled = []
        masks_shuffled = []
        shuffled_indices = np.arange(len(imgs_list))
        random.shuffle(shuffled_indices)
        for ix in shuffled_indices:
            imgs_shuffled.append(imgs_list[ix])
            masks_shuffled.append(masks_list[ix])
        return imgs_shuffled, masks_shuffled
       
    