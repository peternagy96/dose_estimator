#!/usr/bin/env python3
from keras.layers import Input, Conv2D, Conv3D, Activation, Flatten, AveragePooling2D, AveragePooling3D, concatenate
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.engine.topology import Network
from keras.layers.core import Dense
from keras.models import Model

from .layers import ck, c7Ak, dk, Rk, uk, ReflectionPadding2D, IN_LeakyRelu, INLReLU, INLeakyReLU

import tensorflow as tf


class Discriminator(object):
    def __init__(self, name, dim='2D', mode='basic',
                 use_patchgan=True, img_shape=(128, 128, 2)):
        self.img_shape = img_shape
        self.name = name
        self.mode = mode
        self.use_patchgan = use_patchgan
        self.normalization = InstanceNormalization

        D = self.getModel(dim, mode)
        self.summary = D.summary

        # Discriminator builds
        image = Input(shape=self.img_shape)
        guess = D(image)
        self.model = Model(inputs=image, outputs=guess, name=name)

        # Use containers to avoid falsy keras error about weight descripancies
        self.model_static = Network(
            inputs=image, outputs=guess, name=f"{name}_static")

        self.model_static.trainable = False

    def getModel(self, dim, mode):
        if mode == 'basic':
            # 0.5 since we train on real and synthetic images
            self.loss_weights = [0.5]
            if dim == '2D':
                return self.basicDiscriminator()
            elif dim == '3D':
                # return self.basic3DDiscriminator()
                return self.small3DDiscriminator()

        if mode == 'ganimorph':
            # 0.5 since we train on real and synthetic images
            self.loss_weights = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
            if dim == '2D':
                return self.complexDiscriminator()
            elif dim == '3D':
                # return self.basic3DDiscriminator()
                raise NotImplementedError(
                    "3D GaniMorph discriminator model not implemented!")

    def basicDiscriminator(self, name=None):
        # Specify input
        input_img = Input(shape=self.img_shape)
        # Layer 1 (#Instance normalization is not used for this layer)
        x = ck(self.normalization, input_img, 64, False)
        # Layer 2
        x = ck(self.normalization, x, 128, True)
        # Layer 3
        x = ck(self.normalization, x, 256, True)
        # Layer 4
        x = ck(self.normalization, x, 512, True)
        # Output layer
        if self.use_patchgan:
            x = Conv2D(filters=1, kernel_size=4, strides=1, padding='same')(x)
        else:
            x = Flatten()(x)
            x = Dense(1)(x)
        x = Activation('sigmoid')(x)
        return Model(inputs=input_img, outputs=x, name=name)

    def basic3DDiscriminator(self, name=None):
        # Specify input
        input_img = Input(shape=self.img_shape)
        # Layer 1 (#Instance normalization is not used for this layer)
        x = Conv3D(filters=64, kernel_size=3,
                   strides=2, padding='same')(input_img)
        x = LeakyReLU(alpha=0.2)(x)
        # Layer 2
        x = Conv3D(filters=128, kernel_size=3, strides=2, padding='same')(x)
        x = IN_LeakyRelu(x, self.normalization)
        # Layer 3
        x = Conv3D(filters=256, kernel_size=3, strides=2, padding='same')(x)
        x = IN_LeakyRelu(x, self.normalization)
        # Layer 4
        x = Conv3D(filters=512, kernel_size=3, strides=2, padding='same')(x)
        x = IN_LeakyRelu(x, self.normalization)
        # Output layer
        if self.use_patchgan:
            x = Conv3D(filters=1, kernel_size=3, strides=1, padding='same')(x)
        else:
            x = Flatten()(x)
            x = Dense(1)(x)
        x = Activation('sigmoid')(x)
        return Model(inputs=input_img, outputs=x, name=name)

    def complexDiscriminator(self, name=None):
        NF = 32  # channel size
        input_img = Input(shape=self.img_shape)
        l = Conv2D(name='conv0', filters=NF*2, kernel_size=4,
                   strides=2, padding='same')(input_img)
        relu0 = Activation('relu')(l)
        relu1 = Conv2D(name='conv1', filters=NF * 4,
                       kernel_size=4, strides=2, padding='same')(relu0)
        relu1 = INLeakyReLU(relu1, self.normalization)
        relu2 = Conv2D(name='conv2', filters=NF * 8,
                       kernel_size=4, strides=2, padding='same')(relu1)
        relu2 = INLeakyReLU(relu2, self.normalization)

        relu3 = Conv2D(name='convf', filters=NF*8, kernel_size=3,
                       strides=1, padding='same')(relu2)
        atrous = Conv2D(filters=NF*8, kernel_size=3,
                        dilation_rate=2, padding='same')(relu3)
        atrous = INLeakyReLU(atrous, self.normalization)
        atrous2 = Conv2D(filters=NF*8, kernel_size=3,
                         dilation_rate=4, padding='same')(atrous)
        atrous2 = INLeakyReLU(atrous2, self.normalization)
        atrous3 = Conv2D(filters=NF*8, kernel_size=3,
                         dilation_rate=8, padding='same')(atrous2)
        atrous3 = INLeakyReLU(atrous3, self.normalization)
        merge = concatenate([relu3, atrous3], axis=3)
        clean = Conv2D(name='mConv', filters=NF*8, kernel_size=3,
                       strides=1, padding='same')(merge)
        clean = INLeakyReLU(clean, self.normalization)
        lsgan = Conv2D(name='lsconv', filters=1, kernel_size=4, strides=1,
                       padding='same')(clean)  # use_bias=False,
        lsgan = Activation('sigmoid')(lsgan)

        # [lsgan, relu1, relu2, relu3, atrous, atrous2, atrous3, clean]
        return Model(inputs=input_img, outputs=[lsgan, relu1, relu2, relu3, atrous, atrous2, atrous3, clean], name=name)

    def small3DDiscriminator(self, name=None):
        # Specify input
        input_img = Input(shape=self.img_shape)
        # Layer 1 (#Instance normalization is not used for this layer)
        x = Conv3D(filters=32, kernel_size=3,
                   strides=2, padding='same')(input_img)
        x = LeakyReLU(alpha=0.2)(x)
        # Layer 2
        x = Conv3D(filters=64, kernel_size=3, strides=2, padding='same')(x)
        x = IN_LeakyRelu(x, self.normalization)
        # Layer 3
        x = Conv3D(filters=128, kernel_size=3, strides=2, padding='same')(x)
        x = IN_LeakyRelu(x, self.normalization)
        # Layer 4
        x = Conv3D(filters=256, kernel_size=3, strides=2, padding='same')(x)
        x = IN_LeakyRelu(x, self.normalization)
        # Output layer
        if self.use_patchgan:
            x = Conv3D(filters=1, kernel_size=4, strides=1, padding='same')(x)
        else:
            x = Flatten()(x)
            x = Dense(1)(x)
        x = Activation('sigmoid')(x)
        return Model(inputs=input_img, outputs=x, name=name)

    def summary(self):
        return self.model.summary()
