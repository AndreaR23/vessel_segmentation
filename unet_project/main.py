from unet_project.image_utils import ImageUtils
from unet_project.data_augmentation import DataAugmentation
from unet_project.u_net import Unet

import numpy as np
import cv2
from keras.callbacks import EarlyStopping, ModelCheckpoint


def main():
    path_to_imgs = '/home/ajuska/Dokumenty/Skola/diplomka/custom_train/imgs/'
    path_to_masks = '/home/ajuska/Dokumenty/Skola/diplomka/custom_train/masks/'
    img_height = 224
    img_width = 224
    img_channels = 3

    image_utils = ImageUtils(path_to_imgs, path_to_masks, img_height, img_width)
    imgs_masks = image_utils.get_preprocessed_images()

    extend_data = DataAugmentation(input_images=imgs_masks, img_height=img_height,
                                   img_width=img_width, how_many=1)
    extended_imgs, extended_masks = extend_data.extend_database()

    # train_imgs = []
    # for patches in extended_imgs:
    #     if len(patches.shape) > 3:
    #         for patch in patches:
    #             train_imgs.append(patch)
    #     else:
    #         train_imgs.append(patches)
    # print(np.array(train_imgs).shape)
    #
    # train_masks = []
    # for patches in extended_masks:
    #     if patches.shape[0] != img_height:
    #         for patch in patches:
    #             train_masks.append(patch)
    #     else:
    #         train_masks.append(patches)
    #
    # edit_train_masks = []
    # for mask in train_masks:
    #     if len(mask.shape) == 2:
    #         mask = np.expand_dims(mask, axis=-1)
    #     elif len(mask.shape) == 3 and mask.shape[2] == 3:
    #         mask = np.expand_dims(cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY), axis=-1)
    #     else:
    #         mask = mask
    #     edit_train_masks.append(mask)
    # print(np.array(edit_train_masks).shape)
    # # for i in range(0, len(extended_masks)):
    # #     if len(extended_masks[i].shape) == 4:
    # #         mask_ = cv2.cvtColor(np.squeeze(extended_masks[i]), cv2.COLOR_RGB2GRAY)
    # #         extended_masks[i] = np.expand_dims(mask_, axis=-1)
    # #
    # # print(np.array(extended_masks).shape)
    # # print(np.array(extended_imgs).shape)
    # unet = Unet(img_height=img_height, img_width=img_width, img_channels=img_channels)
    # model = unet.create_model()
    # earlystopper = EarlyStopping(patience=5, verbose=1)
    # checkpointer = ModelCheckpoint('model-test.h5', verbose=1, save_best_only=True)
    # model.fit(np.array(train_imgs).astype(np.uint8), np.array(edit_train_masks), validation_split=0.1,
    #           batch_size=1, epochs=10, callbacks=[earlystopper, checkpointer])


if __name__ == '__main__':
    main()
