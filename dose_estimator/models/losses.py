import tensorflow as tf
from keras import backend as K


def lse(y_true, y_pred): # size: (batch, 5, 5, 1)
    loss = tf.reduce_mean(tf.squared_difference(y_pred, y_true))
    return loss

def mae(alpha=0.5): # size: (batch, 80, 80, 2)
    def loss(y_true, y_pred):
        return alpha * tf.reduce_mean(tf.abs(y_pred[...,0] - y_true[...,0])) + (1-alpha) * tf.reduce_mean(tf.abs(y_pred[...,1] - y_true[...,1]))
    return loss

def mae_style(alpha=0.5): # size: (batch, 80, 80, 2)
    def loss(y_true, y_pred):
        return alpha * tf.reduce_mean(tf.abs(y_pred[...,0] - y_true[...,0])) + (1-alpha) * tf.reduce_mean(tf.abs(y_pred[...,1] - y_true[...,1]))
    return loss

def cycle_loss(alpha=0.5): # size: (batch, 80, 80, 2)
    def loss(y_true, y_pred):
        return alpha * tf.reduce_mean(tf.abs(y_pred[...,0] - y_true[...,0])) + (1-alpha) * tf.reduce_mean(tf.abs(y_pred[...,1] - y_true[...,1]))
    return loss

def style_loss(y_true, y_pred):
    total_variation_weight = 1.0
    gram_weight = 1.0
    gm = gm_loss(y_true, y_pred)
    loss = 0.0
    loss += gram_weight  * gm
    loss += total_variation_weight * total_variation_loss(y_pred)
    return loss

def gram_matrix(x):
    assert K.ndim(x) == 4
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (3, 0, 1, 2)))
    gram = K.dot(features, K.transpose(features))
    return gram

def gm_loss(y_true, y_pred):
    assert K.ndim(y_true) == 4
    assert K.ndim(y_pred) == 4
    S = gram_matrix(y_true)
    C = gram_matrix(y_pred)
    channels = 2
    #print(y_pred.shape)
    if y_pred.get_shape().as_list()[1] is None:
        size = 40.0
    else:
        size = y_pred.get_shape().as_list()[1] * y_pred.get_shape().as_list()[2]
    return K.sum(K.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

# the 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent
def total_variation_loss(x):
    assert K.ndim(x) == 4
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