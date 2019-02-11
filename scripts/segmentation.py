import numpy as np
from skimage.io import imshow
import random
from matplotlib import pyplot as plt
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from unet_architecture import unet_architecture
from preprocess_imgs import preprocess_imgs

IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_CHANNELS = 3
TRAIN_PATH_IMGS = '/home/ajuska/Dokumenty/Skola/diplomka/train_Brano/'
TRAIN_PATH_MASKS = '/home/ajuska/Dokumenty/Skola/diplomka/train_Brano/'

model = unet_architecture(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
X_train, Y_train = preprocess_imgs(TRAIN_PATH_IMGS, TRAIN_PATH_MASKS, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)


# Fit model
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-test.h5', verbose=1, save_best_only=True)
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=50,
                    callbacks=[earlystopper, checkpointer])

# Predict on train, val and test
model = load_model('model-test.h5')
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)

# Threshold predictions
preds_train_t = (preds_train > 0.1).astype(np.uint8)
preds_val_t = (preds_val > 0.1).astype(np.uint8)

# Perform a sanity check on some random training/validating samples
ix = random.randint(0, len(preds_train_t))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.show()

ix = random.randint(0, len(preds_val_t))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.show()




























import cv2
import os
from sklearn.model_selection import train_test_split
import keras
import numpy as np

prefix = '/home/ajuska/Dokumenty/Skola/diplomka/segmentation/imgs/'
file_names = []
imgs = []

for file in os.listdir(prefix):
    file_names.append(file)
    img = cv2.imread(os.path.join(prefix, file), 0)
    imgs.append(img)

train_imgs, test_imgs, train_labels, test_labels = train_test_split(imgs, file_names, test_size=0.15)

print('Count of train images: {}'.format(len(train_imgs)))
print('Count of test images: {}'.format(len(test_imgs)))

train_x = np.array(train_imgs) / 255.
# test_x = np.array(test_imgs) / 255.

print(train_x.shape)
print(len(train_labels))
# print(test_x.shape)
print(len(test_labels))
