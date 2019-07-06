#!/usr/bin/env python3
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Conv2D, Conv3D, Activation, concatenate
from keras.models import Model

from .layers import ck, c7Ak, dk, Rk, uk, ReflectionPadding2D, ck3D, c5Ak3D, dk3D, Rk3D, uk3D, ReflectionPadding3D, UnetUpsample, IN_Relu, Unet3dBlock


class Generator(object):
    def __init__(self, name, dim='2D', mode='basic', use_resize_convolution=False,
                 use_identity_learning=True, img_shape=(128, 128, 2)):
        self.img_shape = img_shape
        self.name = name
        self.mode = mode
        self.use_identity_learning = use_identity_learning
        self.normalization = InstanceNormalization
        self.use_resize_convolution = use_resize_convolution

        self.model = self.getModel(dim, mode)

    def getModel(self, dim, mode):
        if mode == 'basic':
            if dim == '2D':
                return self.basicGenerator()
            elif dim == '3D':
                return self.basic3DGenerator()
        elif mode == 'unet':
            if dim == '2D':
                return self.unetGenerator()
            elif dim == '3D':
                return self.unet3DGenerator()
        elif mode == 'small':
            if dim == '2D':
                raise NotImplementedError("2D small generator model not implemented!")
            elif dim == '3D':
                return self.small3DGenerator()


    def basicGenerator(self):
        # Specify input
        input_img = Input(shape=self.img_shape)
        # Layer 1
        x = ReflectionPadding2D((3, 3))(input_img)
        x = c7Ak(self.normalization, x, 48)
        # Layer 2
        x = dk(self.normalization, x, 72)
        # Layer 3
        x = dk(self.normalization, x, 128)

        if self.mode == 'multiscale':
            # Layer 3.5
            x = dk(self.normalization, x, 256)

        # Layer 4-12: Residual layer
        for _ in range(4, 13):
            x = Rk(self.normalization, x)

        if self.mode == 'multiscale':
            # Layer 12.5
            x = uk(self.normalization, self.use_resize_convolution, x, 128)

        # Layer 13
        x = uk(self.normalization, self.use_resize_convolution, x, 72)
        # Layer 14
        x = uk(self.normalization, self.use_resize_convolution, x, 48)
        x = ReflectionPadding2D((3, 3))(x)
        x = Conv2D(self.img_shape[-1], kernel_size=7, strides=1)(x)
        # They say they use Relu but really they do not
        x = Activation('tanh')(x)
        return Model(inputs=input_img, outputs=x, name=self.name)

    def basic3DGenerator(self):
        # Specify input
        input_img = Input(shape=self.img_shape)
        # Layer 1
        x = ReflectionPadding3D((3, 3, 3))(input_img)
        x = c5Ak3D(self.normalization, x, 48)
        # Layer 2
        x = dk3D(self.normalization, x, 72)
        # Layer 3
        x = dk3D(self.normalization, x, 128)

        if self.mode == 'multiscale':
            # Layer 3.5
            x = dk3D(self.normalization, x, 256)

        # Layer 4-12: Residual layer
        for _ in range(4, 9):
            x = Rk3D(self.normalization, x)

        if self.mode == 'multiscale':
            # Layer 12.5
            x = uk3D(self.normalization, self.use_resize_convolution, x, 128)

        # Layer 13
        x = uk3D(self.normalization, self.use_resize_convolution, x, 72)
        # Layer 14
        x = uk3D(self.normalization, self.use_resize_convolution, x, 48)
        x = ReflectionPadding3D((3, 3, 3))(x)
        x = Conv3D(self.img_shape[-1], kernel_size=7, strides=1)(x)
        # They say they use Relu but really they do not
        x = Activation('tanh')(x)
        return Model(inputs=input_img, outputs=x, name=self.name)

    def small3DGenerator(self):
        # Specify input
        input_img = Input(shape=self.img_shape)
        # Layer 1
        x = ReflectionPadding3D((3, 3, 3))(input_img)
        x = c5Ak3D(self.normalization, x, 32)
        # Layer 2
        x = dk3D(self.normalization, x, 64)
        # Layer 3
        x = dk3D(self.normalization, x, 64)

        if self.mode == 'multiscale':
            # Layer 3.5
            x = dk3D(self.normalization, x, 256)

        # Layer 4-12: Residual layer
        for _ in range(4, 9):
            x = Rk3D(self.normalization, x)

        if self.mode == 'multiscale':
            # Layer 12.5
            x = uk3D(self.normalization, self.use_resize_convolution, x, 64)

        # Layer 13
        x = uk3D(self.normalization, self.use_resize_convolution, x, 64)
        # Layer 14
        x = uk3D(self.normalization, self.use_resize_convolution, x, 32)
        x = ReflectionPadding3D((3, 3, 3))(x)
        x = Conv3D(self.img_shape[-1], kernel_size=7, strides=1)(x)
        # They say they use Relu but really they do not
        x = Activation('tanh')(x)
        return Model(inputs=input_img, outputs=x, name=self.name)

    def unetGenerator(self):
        raise NotImplementedError("2D Unet generator not implemented!")

    def unet3DGenerator(self):
        base_filter = 16
        depth = 3
        filters = []
        down_list = []

        channels = self.img_shape[-1]
        input_img = Input(shape=self.img_shape)
        #x = ReflectionPadding3D((3, 3, 3))(input_img)
        x = Conv3D(filters=base_filter, kernel_size=(
                   3, 3, 3), strides=1, padding='same')(input_img)
        x = IN_Relu(x, self.normalization)

        for d in range(depth):
            num_filters = base_filter * (2**d)
            filters.append(num_filters)
            x = Unet3dBlock(x, kernels=(3, 3, 3), n_feat=num_filters, norm=self.normalization)
            down_list.append(x)
            if d != depth - 1:
                x = Conv3D(filters=num_filters*2,
                           kernel_size=(3, 3, 3),
                           strides=(2, 2, 2),
                           padding='same')(x)
                x = IN_Relu(x, self.normalization)

        for d in range(depth-2, -1, -1):
            x = UnetUpsample(x, filters[d], self.normalization)

            x = concatenate([x, down_list[d]], axis=-1)
            x = Conv3D(filters=filters[d],
                       kernel_size=(3, 3, 3),
                       strides=1,
                       padding='same')(x)
            x = IN_Relu(x, self.normalization)
            x = Conv3D(filters=filters[d],
                       kernel_size=(1, 1, 1),
                       strides=1,
                       padding='same')(x)
            x = IN_Relu(x, self.normalization)

        #x = ReflectionPadding3D((3, 3, 3))(x)
        x = Conv3D(filters=channels,
                   kernel_size=(3, 3, 3),
                   padding="same")(x)
        x = Activation('tanh')(x)
        return Model(inputs=input_img, outputs=x, name=self.name)

    def summary(self):
        return self.model.summary()
