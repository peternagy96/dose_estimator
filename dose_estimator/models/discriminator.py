#!/usr/bin/env python3
from keras.layers import Input, Conv2D, Conv3D, Activation, Flatten, AveragePooling2D, AveragePooling3D
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.engine.topology import Network
from keras.layers.core import Dense
from keras.models import Model

from .layers import ck, c7Ak, dk, Rk, uk, ReflectionPadding2D, IN_LeakyRelu


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
                return self.basic3DDiscriminator()

        elif mode == 'multiscale':
            # 0.5 since we train on real and synthetic images
            self.loss_weights = [0.5, 0.5]
            if dim == '2D':
                return self.modelMultiScaleDiscriminator()
            elif dim == '3D':
                return self.model3DMultiScaleDiscriminator()

    def modelMultiScaleDiscriminator(self, name=None):
        x1 = Input(shape=self.img_shape)
        x2 = AveragePooling2D(pool_size=(2, 2))(x1)
        #x4 = AveragePooling2D(pool_size=(2, 2))(x2)

        out_x1 = self.basicDiscriminator('D1')(x1)
        out_x2 = self.basicDiscriminator('D2')(x2)
        #out_x4 = self.modelDiscriminator('D4')(x4)

        return Model(inputs=x1, outputs=[out_x1, out_x2], name=name)

    def model3DMultiScaleDiscriminator(self, name=None):
        x1 = Input(shape=self.img_shape)
        x2 = AveragePooling3D(pool_size=(2, 2))(x1)
        #x4 = AveragePooling2D(pool_size=(2, 2))(x2)

        out_x1 = self.basic3DDiscriminator('D1')(x1)
        out_x2 = self.basic3DDiscriminator('D2')(x2)
        #out_x4 = self.modelDiscriminator('D4')(x4)

        return Model(inputs=x1, outputs=[out_x1, out_x2], name=name)

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
        x = Conv3D(filters=64, kernel_size=3, strides=2, padding='same')(input_img)
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

    def summary(self):
        return self.model.summary()
