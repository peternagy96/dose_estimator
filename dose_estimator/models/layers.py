from keras.layers import Layer, Input, Conv2D, Conv3D, Activation, add, UpSampling2D, UpSampling3D, Conv2DTranspose, Conv3DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras_contrib.layers.normalization.instancenormalization import InputSpec
import tensorflow as tf

# Architecture functions


def ck(norm, x, k, use_normalization):
    x = Conv2D(filters=k, kernel_size=4, strides=2, padding='same')(x)
    # Normalization is not done on the first discriminator layer
    if use_normalization:
        x = norm(axis=3, center=True,
                 epsilon=1e-5)(x, training=True)
    x = LeakyReLU(alpha=0.2)(x)
    return x


def c7Ak(norm, x, k):
    x = Conv2D(filters=k, kernel_size=7, strides=1, padding='valid')(x)
    x = norm(axis=3, center=True, epsilon=1e-5)(x, training=True)
    x = Activation('relu')(x)
    return x


def dk(norm, x, k):
    x = Conv2D(filters=k, kernel_size=3, strides=2, padding='same')(x)
    x = norm(axis=3, center=True, epsilon=1e-5)(x, training=True)
    x = Activation('relu')(x)
    return x


def Rk(norm, x0):
    k = int(x0.shape[-1])
    # first layer
    x = Conv2D(filters=k, kernel_size=3, strides=1, padding='same')(x0)
    x = norm(axis=3, center=True, epsilon=1e-5)(x, training=True)
    x = Activation('relu')(x)
    # second layer
    x = Conv2D(filters=k, kernel_size=3, strides=1, padding='same')(x)
    x = norm(axis=3, center=True, epsilon=1e-5)(x, training=True)
    # merge
    x = add([x, x0])
    return x


def uk(norm, resize, x, k):
    # (up sampling followed by 1x1 convolution <=> fractional-strided 1/2)
    if resize:
        x = UpSampling2D(size=(2, 2))(x)  # Nearest neighbor upsampling
        x = ReflectionPadding2D((1, 1))(x)
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid')(x)
    else:
        x = Conv2DTranspose(filters=k, kernel_size=3, strides=2, padding='same')(
            x)  # this matches fractionally stided with stride 1/2
    x = norm(axis=3, center=True, epsilon=1e-5)(x, training=True)
    x = Activation('relu')(x)
    return x

# 3D basic layers -----------------------------------------------------------

def ck3D(norm, x, k, use_normalization):
    x = Conv3D(filters=k, kernel_size=3, strides=(1,2,2), padding='same')(x)
    # Normalization is not done on the first discriminator layer
    if use_normalization:
        x = norm(axis=4, center=True,
                 epsilon=1e-5)(x, training=True)
    x = LeakyReLU(alpha=0.2)(x)
    return x


def c5Ak3D(norm, x, k):
    x = Conv3D(filters=k, kernel_size=7, strides=1, padding='valid')(x)
    x = norm(axis=4, center=True, epsilon=1e-5)(x, training=True)
    x = Activation('relu')(x)
    return x


def dk3D(norm, x, k):
    x = Conv3D(filters=k, kernel_size=3, strides=(1,2,2), padding='same')(x)
    x = norm(axis=4, center=True, epsilon=1e-5)(x, training=True)
    x = Activation('relu')(x)
    return x


def Rk3D(norm, x0):
    k = int(x0.shape[-1])
    # first layer
    x = Conv3D(filters=k, kernel_size=3, strides=1, padding='same')(x0)
    x = norm(axis=4, center=True, epsilon=1e-5)(x, training=True)
    x = Activation('relu')(x)
    # second layer
    x = Conv3D(filters=k, kernel_size=3, strides=1, padding='same')(x)
    x = norm(axis=4, center=True, epsilon=1e-5)(x, training=True)
    # merge
    x = add([x, x0])
    return x


def uk3D(norm, resize, x, k):
    # (up sampling followed by 1x1 convolution <=> fractional-strided 1/2)
    if resize:
        x = UpSampling3D(size=(2, 2, 2))(x)  # Nearest neighbor upsampling
        x = ReflectionPadding3D((1, 1, 1))(x)
        x = Conv3D(filters=k, kernel_size=3, strides=1, padding='valid')(x)
    else:
        x = Conv3DTranspose(filters=k, kernel_size=3, strides=(1,2,2), padding='same')(
            x)  # this matches fractionally stided with stride 1/2
    x = norm(axis=4, center=True, epsilon=1e-5)(x, training=True)
    x = Activation('relu')(x)
    return x

# 3D Unet layers ------------------------------------------------------------

def Upsample3D(x):
    return UpSampling3D(size=(2, 2, 2))(x)


def UnetUpsample(x, num_filters, norm):
    x = Upsample3D(x)
    x = Conv3D(filters=num_filters,
                         kernel_size=(3, 3, 3),
                         strides=1,
                         padding='same')(x)
    x = IN_Relu(x, norm)
    return x

def IN_Relu(x, norm):
    x = norm(axis=4, center=True, epsilon=1e-5)(x, training=True)
    x = Activation('relu')(x)
    return x

def IN_LeakyRelu(x, norm):
    x = norm(axis=4, center=True, epsilon=1e-5)(x, training=True)
    x = LeakyReLU(alpha=0.2)(x)
    return x


def Unet3dBlock(x, kernels, n_feat, norm, residual=False):
    if residual:
        x_in = x

    for i in range(2):
        x = Conv3D(filters=n_feat,
                   kernel_size=kernels,
                   strides=1,
                   padding='same')(x)
        x = IN_Relu(x, norm)
    return x_in + x if residual else x


# reflection padding taken from
# https://github.com/fastai/courses/blob/master/deeplearning2/neural-style.ipynb
class ReflectionPadding2D(Layer):

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')


class ReflectionPadding3D(Layer):

    def __init__(self, padding=(1, 1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=5)]
        super(ReflectionPadding3D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3] + 2 * self.padding[2], s[4])

    def call(self, x, mask=None):
        w_pad, h_pad, d_pad = self.padding
        return tf.pad(x, [[0, 0], [ h_pad, h_pad], [w_pad, w_pad], [d_pad, d_pad], [0, 0]], 'REFLECT')
