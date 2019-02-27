import numpy as np
import cv2
from scipy.misc import imrotate, imresize
from skimage.transform import resize
import os
from tqdm import tqdm
from keras.preprocessing import image
import imutils
from PIL import Image
from sklearn.feature_extraction.image import extract_patches_2d


class DataAugmentation:
    def __init__(self, input_images_masks, img_height, img_width, how_many=1):
        self._how_many = how_many
        self._input_images_masks = input_images_masks
        self._img_height = img_height
        self._img_width = img_width
        self._images_and_masks = {}
        self._output_images = []
        self._output_masks = []

    def random_horizontal_flip(self, img, mask, p=0.5):
        if p < np.random.rand():
            return img[:, ::-1, :], mask[:, ::-1]
        else:
            return img, mask

    def random_vertical_flip(self, img, mask, p=0.5):
        if p < np.random.rand():
            return img[::-1, :, :], mask[::-1, :]
        else:
            return img, mask

    def random_squared_crop(self, img, mask, minimal_relative_crop_size=0.5):
        h, w, _ = mask.shape

        size = np.random.randint(int(minimal_relative_crop_size * h), h)
        crop_x_origin = np.random.randint(0, h - size + 1)
        crop_y_origin = np.random.randint(0, w - size + 1)
        return img[crop_x_origin: crop_x_origin + size, crop_y_origin: crop_y_origin + size, :], \
               mask[crop_x_origin: crop_x_origin + size, crop_y_origin: crop_y_origin + size]

    def random_rotate(self, img, mask, p=0.5):
        if np.random.rand() > p:
            return img, mask
        else:
            angle = np.random.randint(0, 180)
            return imutils.rotate_bound(img, angle), imutils.rotate_bound(mask, angle)

    def resize(self, img, mask):
        _size = (np.random.randint(4, 2 * img.shape[0]), np.random.randint(4, 2 * img.shape[1]))
        return resize(img, _size, mode='constant', preserve_range=True), \
               resize(mask, _size, mode='constant', preserve_range=True)

    def extend_database(self):
        print('\nExtending database.')

        for key, val in tqdm(self._input_images_masks.items()):
            img = val[0]
            mask = val[1]

            self._output_images.append(img.astype(np.uint8))
            self._output_masks.append(mask)

            for i in range(0, self._how_many):
                horizontal_flipped_img, horizontal_flipped_mask = self.random_horizontal_flip(img, mask)
                vertical_flipped_img, vertical_flipped_mask = self.random_vertical_flip(horizontal_flipped_img,
                                                                                        horizontal_flipped_mask)
                cropped_img, cropped_mask = self.random_squared_crop(vertical_flipped_img,
                                                                     vertical_flipped_mask)
                rotated_img, rotated_mask = self.random_rotate(cropped_img, cropped_mask)
                resized_img, resized_mask = self.resize(rotated_img, rotated_mask)

                self._output_images.append(resized_img.astype(np.uint8))
                self._output_masks.append(resized_mask)

        return self._output_images, self._output_masks





