from unet_project.image_utils import ImageUtils
from unet_project.data_augmentation import DataAugmentation
from unet_project.u_net import Unet

import numpy as np
import cv2
from keras.callbacks import EarlyStopping, ModelCheckpoint


def main():
    path_to_imgs = '/home/ajuska/Dokumenty/Skola/diplomka/custom_train/imgs/'
    path_to_masks = '/home/ajuska/Dokumenty/Skola/diplomka/custom_train/masks/'
    img_height = 448
    img_width = 448
    img_channels = 3

    image_utils = ImageUtils(path_to_imgs, img_height, img_width)
    imgs = image_utils.get_preprocessed_images()

    extend_data = DataAugmentation(input_images=imgs, path_to_masks=path_to_masks, img_height=img_height,
                                   img_width=img_width, how_many=4)
    extended_imgs, extended_masks = extend_data.extend_database()
    for i in range(0, len(extended_masks)):
        if len(extended_masks[i].shape) == 4:
            mask_ = cv2.cvtColor(np.squeeze(extended_masks[i]), cv2.COLOR_RGB2GRAY)
            extended_masks[i] = np.expand_dims(mask_, axis=-1)

    print(np.array(extended_masks).shape)
    print(np.array(extended_imgs).shape)
    unet = Unet(img_height=img_height, img_width=img_width, img_channels=img_channels)
    model = unet.create_model()
    earlystopper = EarlyStopping(patience=5, verbose=1)
    checkpointer = ModelCheckpoint('/models/model-test.h5', verbose=1, save_best_only=True)
    model.fit(np.array(extended_imgs), np.array(extended_masks), validation_split=0.1, batch_size=1, epochs=10,
              callbacks=[earlystopper, checkpointer])


if __name__ == '__main__':
    main()
