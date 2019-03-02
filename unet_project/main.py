from unet_project.image_utils import ImageUtils
from unet_project.data_augmentation import DataAugmentation
from unet_project.create_patches import PatchesCreator
from unet_project.u_net import Unet

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint


def main():
    path_to_imgs = '/home/ajuska/Dokumenty/Skola/diplomka/test_folder/imgs/'
    path_to_masks = '/home/ajuska/Dokumenty/Skola/diplomka/test_folder/masks/'
    img_height = 224
    img_width = 224
    img_channels = 3

    image_utils = ImageUtils(path_to_imgs, path_to_masks, 224, 224)
    imgs_masks = image_utils.get_preprocessed_images()

    extend_data = DataAugmentation(input_images_masks=imgs_masks, how_many=1)
    extended_imgs, extended_masks = extend_data.extend_database()

    patches_creator = PatchesCreator(extended_imgs, extended_masks, img_height, img_width)
    imgs_patches, masks_patches = patches_creator.create_patches()

    unet = Unet(img_height=img_height, img_width=img_width, img_channels=img_channels)
    model = unet.create_model()

    print('Train model.')
    earlystopper = EarlyStopping(patience=5, verbose=1)
    checkpointer = ModelCheckpoint('model-test.h5', verbose=1, save_best_only=True)
    model.fit(np.array(imgs_patches), np.array(masks_patches), validation_split=0.1,
              batch_size=1, epochs=10, callbacks=[earlystopper, checkpointer])


if __name__ == '__main__':
    main()
