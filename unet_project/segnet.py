from keras import models
from keras.layers.core import Activation, Reshape, Permute
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
import json


class SegNet:

    def __init__(self, img_height=320, img_width=320, img_channels=3, classes=2, kernel_size=3):
        self._img_height = img_height
        self._img_width = img_width
        self._img_channels = img_channels
        self._classes = classes
        self._kernel = kernel_size

    def create_model(self):
        img_input = Input(shape=(self._img_heigh, self._img_width, self._img_channels))
        x = img_input
        # Encoder
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(128, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(256, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(512, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # Decoder
        x = Conv2D(512, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(256, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(128, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(classes, (1, 1), padding="valid")(x)
        x = Reshape((self._height * self._img_width, self_classes))(x)
        x = Activation("softmax")(x)
        model = Model(img_input, x)

        optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
        model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])
        model.summary()
        # with open('model_5l_segnet.json', 'w') as outfile:
        #     outfile.write(json.dumps(json.loads(autoencoder.to_json()), indent=2))

        return model
