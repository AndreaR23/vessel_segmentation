from keras import models
from keras.layers.core import Activation, Reshape, Permute
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
import json


class SegNet():

    def __init__(self, img_height=256, img_width=256, img_channels=1, number_labels=2, kernel_size=3):
        self._img_height = img_height
        self._img_width = img_width
        self._img_channels = img_channels
        self._n_labels = number_labels
        self._kernel = kernel_size

    def create_model(self):
        encoding_layers = [
            Conv2D(64,  self._kernel, padding='same', input_shape=(self._img_height, self._img_width, self._img_channels)),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(64,  self._kernel, padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(),

            Conv2D(128, (self._kernel,  self._kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(128, (self._kernel,  self._kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(),

            Conv2D(256, (self._kernel,  self._kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(256, (self._kernel,  self._kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(256, (self._kernel,  self._kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(),

            Conv2D(512, (self._kernel,  self._kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(512, (self._kernel,  self._kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(512, (self._kernel,  self._kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(),

            Conv2D(512, (self._kernel,  self._kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(512, (self._kernel,  self._kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(512, (self._kernel,  self._kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(),
        ]

        autoencoder = models.Sequential()
        autoencoder.encoding_layers = encoding_layers

        for l in autoencoder.encoding_layers:
            autoencoder.add(l)
            print(l.input_shape,l.output_shape,l)

        decoding_layers = [
            UpSampling2D(),
            Conv2D(512, (self._kernel,  self._kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(512, (self._kernel,  self._kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(512, (self._kernel,  self._kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),

            UpSampling2D(),
            Conv2D(512, (self._kernel,  self._kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(512, (self._kernel,  self._kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(256, (self._kernel,  self._kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),

            UpSampling2D(),
            Conv2D(256, (self._kernel,  self._kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(256, (self._kernel,  self._kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(128, (self._kernel,  self._kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),

            UpSampling2D(),
            Conv2D(128, (self._kernel,  self._kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(64, (self._kernel,  self._kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),

            UpSampling2D(),
            Conv2D(64, (self._kernel,  self._kernel), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(self._n_labels, (1, 1), padding='valid'),
            BatchNormalization(),
        ]
        autoencoder.decoding_layers = decoding_layers
        for l in autoencoder.decoding_layers:
            autoencoder.add(l)

        autoencoder.add(Reshape((self._n_labels, self._img_height * self._img_width)))
        autoencoder.add(Permute((2, 1)))
        autoencoder.add(Activation('softmax'))

        optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
        autoencoder.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])

        # with open('model_5l_segnet.json', 'w') as outfile:
        #     outfile.write(json.dumps(json.loads(autoencoder.to_json()), indent=2))

        return autoencoder
