import os
import sys
from tqdm import tqdm
import numpy as np
import cv2
from skimage.transform import resize


def preprocess_imgs(train_path_imgs, train_path_masks, img_height, img_width, img_channels):
    train_names = [filename for filename in os.listdir(os.path.join(train_path_imgs, 'imgs/'))]
    print(len(train_names))

    X_train = np.zeros((len(train_names), img_height, img_width, img_channels), dtype=np.uint8)
    Y_train = np.zeros((len(train_names), img_height, img_width, 1), dtype=np.bool)

    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(train_names), total=len(train_names)):
        path = train_path_imgs
        img = cv2.imread(os.path.join(path, 'imgs', id_))[:, :, :img_channels]
        img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
        X_train[n] = img

        mask = np.zeros((img_height, img_width, 1), dtype=np.bool)
        for mask_file in next(os.walk(os.path.join(train_path_masks, 'masks')))[2]:

            mask = cv2.imread(os.path.join(train_path_masks, 'masks', mask_file))[:, :, 1]
            mask = np.expand_dims(resize(mask, (img_height, img_width), mode='constant',
                                         preserve_range=True), axis=-1)
        Y_train[n] = mask

    print(X_train[0].shape)
    print(Y_train[0].shape)
    return X_train, Y_train

