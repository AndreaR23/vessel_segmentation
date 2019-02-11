import os
import cv2
from skimage.transform import resize
from tqdm import tqdm


class ImageUtils:

    def __init__(self, path_to_dir, img_height, img_width):
        self._path_to_dir = path_to_dir
        self._img_height = img_height
        self._img_width = img_width
        self._loaded_imgs = {}
        self._preprocessed_imgs = {}

    def _load_images(self):
        print('Loading images.')
        for file in tqdm(sorted(os.listdir(self._path_to_dir))):
            img = cv2.imread(os.path.join(self._path_to_dir, file))
            self._loaded_imgs[file] = img

    def _preprocess_images(self):
        print('\nPreprocessing images.')
        for key, val in tqdm(self._loaded_imgs.items()):
            resized_img = resize(val, (self._img_height, self._img_width), mode='constant', preserve_range=True)
            self._preprocessed_imgs[key] = resized_img

    def get_preprocessed_images(self):
        self._load_images()
        self._preprocess_images()
        return self._preprocessed_imgs


# def main():
#     path = '/home/ajuska/Dokumenty/Skola/diplomka/train_Brano/imgs/'
#     img_height = 224
#     img_width = 224
#
#     a = ImageUtils(path, img_height, img_width)
#     imgs = a.get_preprocessed_images()
#     for key, val in imgs.items():
#         print(val.shape)
#         break
#     # print(imgs)
#
#
# if __name__ == '__main__':
#     main()
