import numpy as np
import cv2
from scipy.misc import imrotate, imresize
from skimage.transform import resize
import os
from tqdm import tqdm
from keras.preprocessing import image
import imutils
from PIL import Image


class DataAugmentation:

    def __init__(self, input_images, path_to_masks, img_height, img_width, how_many=1):
        self._how_many = how_many
        self._input_images = input_images
        self._path_to_masks = path_to_masks
        self._img_height = img_height
        self._img_width = img_width
        self._images_and_masks = {}
        self._output_images = []
        self._output_masks = []

    def _load_masks(self):
        for key, val in tqdm(self._input_images.items()):
            mask_ = cv2.imread(os.path.join(self._path_to_masks, key))[:, :, 1]
            mask = resize(mask_, (self._img_height, self._img_width), mode='constant', preserve_range=True)

            if mask is not None:
                array_mask = image.img_to_array(mask)
                self._images_and_masks[key] = (val, array_mask)
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
        h, w, _ = mask.shape

        size = np.random.randint(int(minimal_relative_crop_size * h), h)
        crop_x_origin = np.random.randint(0, h - size + 1)
        crop_y_origin = np.random.randint(0, w - size + 1)
        return img[crop_x_origin: crop_x_origin + size, crop_y_origin: crop_y_origin + size, :], \
               mask[crop_x_origin: crop_x_origin + size, crop_y_origin: crop_y_origin + size]

    def _random_rotate(self, img, mask, p=0.5):
        if np.random.rand() > p:
            return img, mask
        else:
            angle = np.random.randint(0, 180)
            # return imrotate(img, angle, interp="bilinear"), imrotate(mask, angle)
            return imutils.rotate_bound(img, angle), imutils.rotate_bound(mask, angle)

    def _resize(self, img, mask):
        _size = (np.random.randint(10, 2*img.shape[0]), np.random.randint(10, 2*img.shape[1]))
        return resize(img, _size, mode='constant', preserve_range=True), \
               resize(mask, _size, mode='constant', preserve_range=True)

    def _change_dims(self, mask_or_img, mode, img_heigh=448, img_width=448, fill_color=(0, 0, 0)):
        if mode == 'mask':
            set_mode = '1'
            if len(mask_or_img) == 2:
                x, y = mask_or_img.shape
            else:
                mask_or_img = np.squeeze(mask_or_img)
                x, y = mask_or_img.shape

        elif mode == 'img':
            set_mode = 'RGB'
            x, y, _ = mask_or_img.shape

        if x < img_heigh or y < img_width:
            desired_ratio = img_width / img_heigh
            w = max(img_width, x)
            h = int(w / desired_ratio)
            if h < y:
                h = y
                w = int(h * desired_ratio)
            im = Image.fromarray(mask_or_img.astype('uint8'), mode=set_mode)
            new_im = Image.new('RGB', (w, h), fill_color)
            new_im.paste(im, ((w - x) // 2, (h - y) // 2))
            return np.array(new_im.resize((img_width, img_heigh)))

        elif x > img_heigh or y > img_width:
            left = (y - img_width) / 2
            top = (x - img_heigh) / 2
            right = (y + img_width) / 2
            bottom = (x + img_heigh) / 2

            im = Image.fromarray(mask_or_img.astype('uint8'), mode=set_mode)
            return np.array(im.crop((left, top, right, bottom)))

        else:
            return mask_or_img

    def extend_database(self):
        print('\nExtending database.')
        DataAugmentation._load_masks(self)

        for key, val in tqdm(self._images_and_masks.items()):
            img = val[0]
            mask = val[1]

            result_img = DataAugmentation._change_dims(self, img, mode='img')
            result_mask = DataAugmentation._change_dims(self, mask, mode='mask')
            self._output_images.append(result_img)
            self._output_masks.append(np.expand_dims(result_mask, axis=-1))

            for i in range(0, self._how_many):
                horizontal_flipped_img, horizontal_flipped_mask = DataAugmentation._random_horizontal_flip(self, img, mask)
                vertical_flipped_img, vertical_flipped_mask = DataAugmentation._random_vertical_flip(self,
                                                                                                     horizontal_flipped_img,
                                                                                                     horizontal_flipped_mask)
                cropped_img, cropped_mask = DataAugmentation._random_squared_crop(self, vertical_flipped_img,
                                                                                  vertical_flipped_mask)
                rotated_img, rotated_mask = DataAugmentation._random_rotate(self, cropped_img, cropped_mask)
                resized_img, resized_mask = DataAugmentation._resize(self, rotated_img, rotated_mask)

                # try:
                result_img = DataAugmentation._change_dims(self, resized_img, mode='img')
                result_mask = DataAugmentation._change_dims(self, resized_mask, mode='mask')

                self._output_images.append(result_img)
                self._output_masks.append(np.expand_dims(result_mask, axis=-1))

        return self._output_images, self._output_masks





