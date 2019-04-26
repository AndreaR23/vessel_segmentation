import random
import numpy as np
import os
from scipy.signal import medfilt2d
import cv2

from classes.image_utils import ImageUtils
from segmentation_models.backbones import get_preprocessing


class ProcessResults:

    def __init__(self):
        self._imgs = []
        self._masks = []
        self._predicted_imgs = []

    def predict_images_from_dir(self, backbone, val_frame_path, val_mask_path, model, img_height, img_width, 
                                preprocess=True, how_many=20):

        preprocess_input = get_preprocessing(backbone)

        names = os.listdir(val_frame_path)
        random.shuffle(names)
        
        for idx in range(how_many):
            image_ut = ImageUtils(val_frame_path, val_mask_path, img_height, img_width)
            
            loaded_img, loaded_mask = image_ut.load_image_mask_pair(names[idx])
            resized_img, resized_mask = image_ut.preprocess_image_mask_pair(loaded_img, loaded_mask)
            
            if preprocess:
                test_img = preprocess_input(resized_img)/255
            else:
                test_img = resized_img/255
            test_mask = resized_mask/255
            
            expand_test = np.expand_dims(test_img, axis=0)
            predicted_img = model.predict(expand_test)
            
            self._imgs.append(test_img)
            self._masks.append(test_mask)
            self._predicted_imgs.append(predicted_img)

    @staticmethod
    def predict_image(img, model, backbone, preprocess=True):
        preprocess_input = get_preprocessing(backbone)
        if preprocess:
                img = preprocess_input(img)/255
        else:
            img = img/255
            
        expand_test = np.expand_dims(img, axis=0)
        predicted_img = model.predict(expand_test)
        return predicted_img
        
    @staticmethod
    def med_ext_med_filter(predicted_img, med1_kernel_size=3, ext_kernel_size=(12,12), med2_kernel_size=7):
        kernel = np.ones(ext_kernel_size, np.uint8)
        img_squeezed = np.squeeze(predicted_img)   
        
        med1_filter_img = medfilt2d(img_squeezed, med1_kernel_size)
        ext_filter = cv2.morphologyEx(med1_filter_img, cv2.MORPH_CLOSE, kernel)
        med2_filter_img = medfilt2d(ext_filter, med2_kernel_size)
        
        return med2_filter_img

    @staticmethod
    def tresholding(predicted_img, treshold=0.9):
        predicted_img[predicted_img > treshold] = 1
        predicted_img[predicted_img > treshold] = 1
        gauss_filter_img = cv2.GaussianBlur(predicted_img, (5,5), 0)

        return gauss_filter_img



    