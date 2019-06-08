#!/usr/bin/env python3
from keras.layers import Input, Conv2D, Activation
from keras.models import Model

import layers


class Generator(object):
    def __init__(self, name, mode='basic',
                 use_identity_learning=True, img_shape=(128, 128, 2)):
        self.img_shape = img_shape
        self.name = name
        self.mode = mode
        self.use_identity_learning = use_identity_learning

        self.model = self.getModel(mode)

    def getModel(self, mode):
        if mode == 'basic':
            return self.basicGenerator()

    def basicGenerator(self):
        # Specify input
        input_img = Input(shape=self.img_shape)
        # Layer 1
        x = layers.ReflectionPadding2D((3, 3))(input_img)
        x = layers.c7Ak(x, 48)
        # Layer 2
        x = layers.dk(x, 72)
        # Layer 3
        x = layers.dk(x, 128)

        if self.mode == 'multiscale':
            # Layer 3.5
            x = layers.dk(x, 256)

        # Layer 4-12: Residual layer
        for _ in range(4, 13):
            x = layers.Rk(x)

        if self.mode == 'multiscale':
            # Layer 12.5
            x = layers.uk(x, 128)

        # Layer 13
        x = layers.uk(x, 72)
        # Layer 14
        x = layers.uk(x, 48)
        x = layers.ReflectionPadding2D((3, 3))(x)
        x = Conv2D(self.img_shape[-1], kernel_size=7, strides=1)(x)
        # They say they use Relu but really they do not
        x = Activation('tanh')(x)
        return Model(inputs=input_img, outputs=x, name=self.name)

    def summary(self):
        return self.model.summary()
