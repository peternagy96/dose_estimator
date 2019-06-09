from keras.layers import Layer, Input, Conv2D, Activation, add, UpSampling2D, Conv2DTranspose
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
