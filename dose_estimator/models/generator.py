#!/usr/bin/env python3
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Conv2D, Conv3D, Conv2DTranspose, Activation, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model

from .layers import ck, ck_rl, dck, c7Ak, dk, Rk, uk, ReflectionPadding2D, ck3D, c5Ak3D, dk3D, Rk3D, uk3D, ReflectionPadding3D, UnetUpsample, IN_Relu, Unet3dBlock
from tensorpack import LinearWrap


class Generator(object):
    def __init__(self, name, dim='2D', mode='basic', use_resize_convolution=False,
                 use_identity_learning=True, img_shape=(128, 128, 2), style_loss=False):
        self.img_shape = img_shape
        self.name = name
        self.mode = mode
        self.use_identity_learning = use_identity_learning
        self.normalization = InstanceNormalization
        self.use_resize_convolution = use_resize_convolution

        self.model = self.getModel(dim, mode, style_loss)

    def getModel(self, dim, mode, style_loss):
        if mode == 'basic':
            if style_loss:
                if dim == '2D':
                    return self.styleGenerator()
                elif dim == '3D':
                    return self.style3DGenerator()
            else:
                if dim == '2D':
                    return self.basicGenerator()
                elif dim == '3D':
                    return self.basic3DGenerator()
        elif mode == 'ganimorph':
            if dim == '2D':
                return self.complexGenerator()
            elif dim == '3D':
                raise NotImplementedError(
                    "3D GaniMorph generator model not implemented!")

    def basicGenerator(self, pool=False):
        # Specify input
        input_img = Input(shape=self.img_shape)
        # Layer 1
        x = ReflectionPadding2D((3, 3))(input_img)
        x = c7Ak(self.normalization, x, 48, name='c7ak_1')
        # Layer 2
        x = dk(self.normalization, x, 72, pool=pool, name='dk_1')
        # Layer 3
        x = dk(self.normalization, x, 128, pool=pool, name='dk_2')

        # Layer 4-12: Residual layer
        for i in range(4, 13):
            x = Rk(self.normalization, x, name=f"res_{i}", style=False)

        # Layer 13
        x = uk(self.normalization, self.use_resize_convolution,
               x, 72, pool=pool, name='uk_1')
        # Layer 14
        x = uk(self.normalization, self.use_resize_convolution,
               x, 48, pool=pool, name='uk_2')
        x = ReflectionPadding2D((3, 3))(x)
        x = Conv2D(self.img_shape[-1], kernel_size=7,
                   strides=1, name='final')(x)
        # They say they use Relu but really they do not
        x = LeakyReLU(alpha=0.2)(x)
        #x = Activation('tanh')(x)
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

        # Layer 4-12: Residual layer
        for _ in range(4, 13):
            x = Rk3D(self.normalization, x, style=False)

        # Layer 13
        x = uk3D(self.normalization, self.use_resize_convolution, x, 72)
        # Layer 14
        x = uk3D(self.normalization, self.use_resize_convolution, x, 48)
        x = ReflectionPadding3D((3, 3, 3))(x)
        x = Conv3D(self.img_shape[-1], kernel_size=7, strides=1)(x)
        # They say they use Relu but really they do not
        x = Activation('tanh')(x)
        return Model(inputs=input_img, outputs=x, name=self.name)

    def complexGenerator(self, name=None):
        def res_group(inp, name, depth, filters):
            x = inp
            for k in range(depth):
                x0 = x
                x = Conv2D(name=f"{name}_{k}0", filters=filters,
                           strides=1, kernel_size=3, padding='same')(x)
                x = self.normalization(
                    axis=3, center=True, epsilon=1e-5)(x, training=True)
                x = Activation('relu')(x)
                x = Conv2D(name=f"{name}_{k}1", filters=filters,
                           strides=1, kernel_size=3, padding='same')(x)
                x = self.normalization(
                    axis=3, center=True, epsilon=1e-5)(x, training=True)
                x = Activation('relu')(x)
                x = concatenate([x, x0], axis=3)
                x = Conv2D(name=f"{name}_{k}2", filters=filters,
                           strides=1, kernel_size=3, padding='same')(x)
                x = self.normalization(
                    axis=3, center=True, epsilon=1e-5)(x, training=True)
                x = Activation('relu')(x)
            return x

        NF = 32
        subDepth = 3
        input_img = Input(shape=self.img_shape)
        conv0 = ck(self.normalization, input_img, NF, True)
        conv1 = ck(self.normalization, conv0, NF * 2, True)
        layer1 = res_group(conv1, 'layer1', subDepth, NF*2)
        conv2 = ck(self.normalization, layer1, NF * 4, True)
        layer2 = res_group(conv2, 'layer2', subDepth, NF*4)
        conv3 = ck(self.normalization, layer2, NF * 8, True)
        l = res_group(conv3, 'layer3', subDepth, NF*8)
        deconv0 = dck(self.normalization, l, NF * 4, True)
        up1 = concatenate([deconv0, layer2], axis=3)
        b_layer_2 = res_group(up1, 'blayer2', subDepth, NF * 4)
        deconv1 = dck(self.normalization, b_layer_2, NF * 2, True)
        up2 = concatenate([deconv1, layer1], axis=3)
        b_layer_1 = res_group(up2, 'blayer1', subDepth, NF * 2)
        deconv2 = dck(self.normalization, b_layer_1, NF, True)
        deconv3 = Conv2DTranspose(
            name='deconv3', filters=self.img_shape[-1], kernel_size=4, strides=2, padding='same', activation='tanh')(deconv2)
        return Model(inputs=input_img, outputs=deconv3, name=self.name)

    def styleGenerator(self):
        # Specify input
        input_img = Input(shape=self.img_shape)
        # Layer 1
        x = ReflectionPadding2D((3, 3))(input_img)
        x = c7Ak(self.normalization, x, 48, name='c7ak_1')
        # Layer 2
        x = dk(self.normalization, x, 72, name='dk_1')
        # Layer 3
        x = dk(self.normalization, x, 128, name='dk_2')

        # Layer 4-12: Residual layer
        res = []
        for i in range(4, 13):
            x, x0 = Rk(self.normalization, x, name=f"res_{i}", style=True)
            res.append(x0)

        # Layer 13
        x = uk(self.normalization, self.use_resize_convolution, x, 72, name='uk_1')
        # Layer 14
        x = uk(self.normalization, self.use_resize_convolution, x, 48, name='uk_2')
        x = ReflectionPadding2D((3, 3))(x)
        x = Conv2D(self.img_shape[-1], kernel_size=7,
                   strides=1, name='final')(x)
        x = Activation('tanh')(x)
        res.insert(0, x)
        return Model(inputs=input_img, outputs=res, name=self.name)

    def style3DGenerator(self):
        # Specify input
        input_img = Input(shape=self.img_shape)
        # Layer 1
        x = ReflectionPadding3D((3, 3, 3))(input_img)
        x = c5Ak3D(self.normalization, x, 48)
        # Layer 2
        x = dk3D(self.normalization, x, 72)
        # Layer 3
        x = dk3D(self.normalization, x, 128)

        # Layer 4-12: Residual layer
        res = []
        for _ in range(4, 9):
            x, x0 = Rk3D(self.normalization, x, style=True)
            res.append(x0)

        # Layer 13
        x = uk3D(self.normalization, self.use_resize_convolution, x, 72)
        # Layer 14
        x = uk3D(self.normalization, self.use_resize_convolution, x, 48)
        x = ReflectionPadding3D((3, 3, 3))(x)
        x = Conv3D(self.img_shape[-1], kernel_size=7, strides=1)(x)
        # They say they use Relu but really they do not
        x = Activation('tanh')(x)
        res.insert(0, x)
        return Model(inputs=input_img, outputs=res, name=self.name)

    def summary(self):
        return self.model.summary()
