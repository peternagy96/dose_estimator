import tensorflow as tf
from keras import backend as K
import keras_contrib.backend as KC
from keras.layers import Conv2D


def lse(y_true, y_pred):  # size: (batch, 5, 5, 1)
    loss = tf.reduce_mean(tf.squared_difference(y_pred, y_true))
    return loss


def mae(alpha=0.5):  # size: (batch, 80, 80, 2)
    def loss(y_true, y_pred):
        return alpha * tf.reduce_mean(tf.abs(y_pred[..., 0] - y_true[..., 0])) + (1-alpha) * tf.reduce_mean(tf.abs(y_pred[..., 1] - y_true[..., 1]))
    return loss


def cycle_loss(alpha=0.5, ssim=False, crop=True):  # size: (batch, 80, 80, 2)
    def loss(y_true, y_pred):
        x = alpha * tf.reduce_mean(tf.abs(y_pred[..., 0] - y_true[..., 0])) + (
            1-alpha) * tf.reduce_mean(tf.abs(y_pred[..., 1] - y_true[..., 1]))
        if ssim:
            x += dssim(y_true, y_pred, crop)
        return x
    return loss


def s_loss(crop=True):
    def loss(y_true, y_pred):
        total_variation_weight = 0.5
        gram_weight = 0.5
        gm = gm_loss(y_true, y_pred, crop)
        l = gram_weight * gm
        l += total_variation_weight * total_variation_loss(y_pred, crop)
        return l
    return loss


def gram_matrix(x):
    assert K.ndim(x) == 4
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (3, 0, 1, 2)))
    gram = K.dot(features, K.transpose(features))
    return gram


def gm_loss(crop=True):
    def loss(y_true, y_pred):
        if K.ndim(y_true) == 5:
            if crop:
                y_true = K.reshape(y_true, [-1, 80, 80, 2])
                y_pred = K.reshape(y_pred, [-1, 80, 80, 2])
            else:
                y_true = K.reshape(y_true, [-1, 128, 128, 2])
                y_pred = K.reshape(y_pred, [-1, 128, 128, 2])
        assert K.ndim(y_true) == 4
        assert K.ndim(y_pred) == 4
        S = gram_matrix(y_true)
        C = gram_matrix(y_pred)
        channels = 2
        # print(y_pred.shape)
        if y_pred.get_shape().as_list()[1] is None:
            size = 400.0
        else:
            size = y_pred.get_shape().as_list(
            )[1] * y_pred.get_shape().as_list()[2]
        return K.sum(K.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))
    return loss

# designed to keep the generated image locally coherent


def total_variation_loss(x, crop):
    if K.ndim(x) == 5:
        if crop:
            x = K.reshape(x, [-1, 80, 80, 2])
        else:
            x = K.reshape(x, [-1, 128, 128, 2])
    assert K.ndim(x) == 4
    if x.get_shape().as_list()[1] is None:
        img_nrows = 20
        img_ncols = 20
    else:
        img_nrows = int(x.get_shape().as_list()[1])
        img_ncols = int(x.get_shape().as_list()[2])
    if K.image_data_format() == 'channels_first':
        a = K.square(
            x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
        b = K.square(
            x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
    else:
        a = K.square(
            x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
        b = K.square(
            x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


def null_loss(y_true, y_pred):
    return tf.Variable(0.0)


def feature_match_loss(y_true, y_pred):
    loss = tf.reduce_mean(tf.squared_difference(
        tf.reduce_mean(y_true, 0),
        tf.reduce_mean(y_pred, 0)))
    return loss


def dssim(y_true, y_pred, crop):
    if len(K.int_shape(y_true)) == 5:
        if crop:
            y_true = K.reshape(y_true, [-1, 80, 80, 2])
            y_pred = K.reshape(y_pred, [-1, 80, 80, 2])
        else:
            y_true = K.reshape(y_true, [-1, 128, 128, 2])
            y_pred = K.reshape(y_pred, [-1, 128, 128, 2])
    kernel_size = 3
    k1 = 0.01
    k2 = 0.03
    max_value = 1.0
    c1 = (k1 * max_value) ** 2
    c2 = (k2 * max_value) ** 2
    dim_ordering = K.image_data_format()

    kernel = [kernel_size, kernel_size]
    y_true = K.reshape(y_true, [-1] + list(K.int_shape(y_pred)[1:]))
    y_pred = K.reshape(y_pred, [-1] + list(K.int_shape(y_pred)[1:]))

    patches_pred = KC.extract_image_patches(y_pred, kernel, kernel, 'valid',
                                            dim_ordering)
    patches_true = KC.extract_image_patches(y_true, kernel, kernel, 'valid',
                                            dim_ordering)

    # Reshape to get the var in the cells
    bs, w, h, c1, c2, c3 = K.int_shape(patches_pred)
    patches_pred = K.reshape(patches_pred, [-1, w, h, c1 * c2 * c3])
    patches_true = K.reshape(patches_true, [-1, w, h, c1 * c2 * c3])
    # Get mean
    u_true = K.mean(patches_true, axis=-1)
    u_pred = K.mean(patches_pred, axis=-1)
    # Get variance
    var_true = K.var(patches_true, axis=-1)
    var_pred = K.var(patches_pred, axis=-1)
    # Get std dev
    covar_true_pred = K.mean(
        patches_true * patches_pred, axis=-1) - u_true * u_pred

    ssim = (2 * u_true * u_pred + c1) * (2 * covar_true_pred + c2)
    denom = ((K.square(u_true)
              + K.square(u_pred)
              + c1) * (var_pred + var_true + c2))
    ssim /= denom  # no need for clipping, c1 and c2 make the denom non-zero
    return K.mean((1.0 - ssim) / 2.0)
