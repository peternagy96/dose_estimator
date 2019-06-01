#!/usr/bin/env python3
import sys
import numpy as np
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

import layers


class Discriminator(object):
    def __init__(self, use_multiscale_discriminator=False, img_shape=(128,128,2)):
        self.img_shape = img_shape

        if use_multiscale_discriminator:
            D_A = self.modelMultiScaleDiscriminator()
            D_B = self.modelMultiScaleDiscriminator()
            loss_weights_D = [0.5, 0.5] # 0.5 since we train on real and synthetic images
        else:
            D_A = self.modelDiscriminator()
            D_B = self.modelDiscriminator()
            loss_weights_D = [0.5]  # 0.5 since we train on real and synthetic images
        # D_A.summary()

        # Discriminator builds
        image_A = Input(shape=self.img_shape)
        image_B = Input(shape=self.img_shape)
        guess_A = D_A(image_A)
        guess_B = D_B(image_B)
        self.D_A = Model(inputs=image_A, outputs=guess_A, name='D_A_model')
        self.D_B = Model(inputs=image_B, outputs=guess_B, name='D_B_model')

        # self.D_A.summary()
        # self.D_B.summary()
        self.D_A.compile(optimizer=self.opt_D,
                         loss=self.lse,
                         loss_weights=loss_weights_D)
        self.D_B.compile(optimizer=self.opt_D,
                         loss=self.lse,
                         loss_weights=loss_weights_D)

        # Use containers to avoid falsy keras error about weight descripancies
        self.D_A_static = Network(inputs=image_A, outputs=guess_A, name='D_A_static_model')
        self.D_B_static = Network(inputs=image_B, outputs=guess_B, name='D_B_static_model')


    def modelMultiScaleDiscriminator(self, name=None):
        x1 = Input(shape=self.img_shape)
        x2 = AveragePooling2D(pool_size=(2, 2))(x1)
        #x4 = AveragePooling2D(pool_size=(2, 2))(x2)

        out_x1 = self.modelDiscriminator('D1')(x1)
        out_x2 = self.modelDiscriminator('D2')(x2)
        #out_x4 = self.modelDiscriminator('D4')(x4)

        return Model(inputs=x1, outputs=[out_x1, out_x2], name=name)

    def modelDiscriminator(self, name=None):
        # Specify input
        input_img = Input(shape=self.img_shape)
        # Layer 1 (#Instance normalization is not used for this layer)
        x = self.ck(input_img, 64, False)
        # Layer 2
        x = self.ck(x, 128, True)
        # Layer 3
        x = self.ck(x, 256, True)
        # Layer 4
        x = self.ck(x, 512, True)
        # Output layer
        if self.use_patchgan:
            x = Conv2D(filters=1, kernel_size=4, strides=1, padding='same')(x)
        else:
            x = Flatten()(x)
            x = Dense(1)(x)
        x = Activation('sigmoid')(x)
        return Model(inputs=input_img, outputs=x, name=name)

    def summary(self):
        return self.Discriminator.summary()

