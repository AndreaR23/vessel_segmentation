import os
import cv2
from tqdm import tqdm


class ImageUtils:
    def __init__(self, path_to_imgs_dir, path_to_masks_dir, img_height, img_width):
        self._path_to_imgs_dir = path_to_imgs_dir
        self._path_to_masks_dir = path_to_masks_dir
        self._img_height = img_height
        self._img_width = img_width
        self._loaded_imgs = {}
        self._loaded_masks = {}
        self._preprocessed_imgs_masks = {}

    def _load_images(self):
        print('Loading images.')
        for file in tqdm(sorted(os.listdir(self._path_to_imgs_dir))):
            img = cv2.imread(os.path.join(self._path_to_imgs_dir, file))[:, :, :3]
            self._loaded_imgs[file] = img

    def _load_masks(self):
        print('\nLoading masks.')
        for key, val in self._loaded_imgs.items():
            path = os.path.join(self._path_to_masks_dir, key)
            mask = cv2.imread(os.path.join(self._path_to_masks_dir, key))[:, :, :1]
            self._loaded_masks[key] = mask

    def _preprocess_images(self):
        print('\nPreprocessing images.')
        for key, val in tqdm(self._loaded_imgs.items()):
            img_ = cv2.cvtColor(val[50:50+670, 50:50+900], cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(cv2.equalizeHist(img_), cv2.COLOR_GRAY2RGB)
            mask_ = self._loaded_masks[key]
            mask = mask_[50:50+670, 50:50+900]
            self._preprocessed_imgs_masks[key] = (img, mask)

    def get_preprocessed_images(self):
        self._load_images()
        self._load_masks()
        self._preprocess_images()
        return self._preprocessed_imgs_masks
