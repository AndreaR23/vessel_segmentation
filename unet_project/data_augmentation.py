import numpy as np
import cv2
from scipy.misc import imrotate, imresize
from skimage.transform import resize
import os
from tqdm import tqdm
from random import  uniform

from project.image_utils import ImageUtils


class DataAugmentation:

    def __init__(self, input_images, path_to_masks, img_height, img_width, how_many=10):
        self._how_many = how_many
        self._input_images = input_images
        self._path_to_masks = path_to_masks
        self._img_height = img_height
        self._img_width = img_width
        self._images_and_masks = {}
        self._output_images = []

    def _load_masks(self):
        for key, val in tqdm(self._input_images.items()):
            mask_ = cv2.imread(os.path.join(self._path_to_masks, key), 0)
            mask = resize(mask_, (self._img_height, self._img_width), mode='constant', preserve_range=True)

            if mask is not None:
                self._images_and_masks[key] = (val, mask)
        return self._images_and_masks

    def _random_horizontal_flip(self, img, mask, p=0.5):
        if p < np.random.rand():
            return img[:, ::-1, :], mask[:, ::-1]
        else:
            return img, mask

    def _random_vertical_flip(self, img, mask, p=0.5):
        if p < np.random.rand():
            return img[::-1, :, :], mask[::-1, :]
        else:
            return img, mask

    def _random_squared_crop(self, img, mask, minimal_relative_crop_size=0.5):
        h, w = mask.shape

        size = np.random.randint(int(minimal_relative_crop_size * h), h)
        print(size)
        crop_x_origin = np.random.randint(0, h - size + 1)
        crop_y_origin = np.random.randint(0, w - size + 1)
        return img[crop_x_origin: crop_x_origin + size, crop_y_origin: crop_y_origin + size, :], \
               mask[crop_x_origin: crop_x_origin + size, crop_y_origin: crop_y_origin + size]

    def _random_rotate(self, img, mask, p=0.5):
        if np.random.rand() > p:
            return img, mask
        else:
            angle = np.random.randint(0, 180)
            return imrotate(img, angle, interp="bilinear"), imrotate(mask, angle)

    def _resize(self, img, mask, _range):
        _size = uniform(_range[0], _range[1])
        return imresize(img, _size), imresize(mask, _size)

    def extend_database(self):
        DataAugmentation._load_masks(self)

        for key, val in self._images_and_masks.items():
            print(key)
            img = val[0]
            mask = val[1]
            self._output_images.append([img, mask])

            for i in range(0, self._how_many):
                horizontal_flipped_img, horizontal_flipped_mask = DataAugmentation._random_horizontal_flip(self, img,
                                                                                                           mask)
                vertical_flipped_img, vertical_flipped_mask = DataAugmentation._random_vertical_flip(self,
                                                                                                     horizontal_flipped_img,
                                                                                                     horizontal_flipped_mask)
                cropped_img, cropped_mask = DataAugmentation._random_squared_crop(self, vertical_flipped_img,
                                                                                  vertical_flipped_mask)
                rotated_img, rotated_mask = DataAugmentation._random_rotate(self, cropped_img, cropped_mask)
                resized_img, resized_mask = DataAugmentation._resize(self, rotated_img, rotated_mask, [0.5, 1.5])
                self._output_images.append([resized_img, resized_mask])

        return self._output_images


def main():
    path_to_imgs = '/home/ajuska/Dokumenty/Skola/diplomka/custom_train/imgs/'
    path_to_masks = '/home/ajuska/Dokumenty/Skola/diplomka/custom_train/masks/'
    img_height = 700
    img_width = 950

    a = ImageUtils(path_to_imgs, img_height, img_width)
    imgs = a.get_preprocessed_images()

    extend = DataAugmentation(input_images=imgs, path_to_masks=path_to_masks, img_height=img_height,
                              img_width=img_width, how_many=4)
    ext_imgs = extend.extend_database()
    # b = extend.loadMasks()
    print(len(ext_imgs))


if __name__ == '__main__':
   main()


