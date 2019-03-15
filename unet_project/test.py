from keras.callbacks import EarlyStopping, ModelCheckpoint

from image_utils import ImageUtils
#from data_augmentation import DataAugmentation
from segnet import SegNet
#from create_patches import PatchesCreator
import cv2
import numpy as np

path_to_imgs = '/home/ajuska/Dokumenty/Skola/diplomka/disk_data/imgs/'
path_to_masks = '/home/ajuska/Dokumenty/Skola/diplomka/disk_data/masks/'
img_height = 320
img_width = 320
n_labels = 1
kernel = 3
img_channels = 1

image_utils = ImageUtils(path_to_imgs, path_to_masks, img_height, img_width)
imgs_masks = image_utils.get_preprocessed_images()

# data_augmentor = DataAugmentation(imgs_masks, how_many=1)
# aug_imgs, aug_masks = data_augmentor.extend_database()

imgs = []
masks = []
for key, val in imgs_masks.items():
    imgs.append(val[0])
    masks.append(val[1])

trainX = [cv2.resize(x, (img_height, img_width)) for x in imgs]
trainY = [cv2.resize(x, (img_height, img_width)) for x in masks]

trainX = np.array([cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_BGR2GRAY) for x in trainX])
trainX = np.array([np.expand_dims(x/255, axis=-1) for x in trainX])

trainY = np.reshape(trainY,(len(trainY),img_height*img_width,img_channels))

segnet = SegNet(img_height=img_height, img_width=img_width, img_channels=img_channels,
                number_labels=n_labels, kernel_size=kernel)
model = segnet.create_model()

earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-test.h5', verbose=1, save_best_only=True)
results = model.fit(trainX, trainY, validation_split=0.1, batch_size=128, epochs=50, verbose=1,
                    callbacks=[earlystopper, checkpointer])