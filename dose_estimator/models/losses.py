import tensorflow as tf
from keras import backend as K


def lse(y_true, y_pred): # size: (batch, 5, 5, 1)
    loss = tf.reduce_mean(tf.squared_difference(y_pred, y_true))
    return loss

def mae(alpha=0.5): # size: (batch, 80, 80, 2)
    def loss(y_true, y_pred):
        return alpha * tf.reduce_mean(tf.abs(y_pred[...,0] - y_true[...,0])) + (1-alpha) * tf.reduce_mean(tf.abs(y_pred[...,1] - y_true[...,1]))
    return loss

def cycle_loss(alpha=0.5): # size: (batch, 80, 80, 2)
    def loss(y_true, y_pred):
        return alpha * tf.reduce_mean(tf.abs(y_pred[...,0] - y_true[...,0])) + (1-alpha) * tf.reduce_mean(tf.abs(y_pred[...,1] - y_true[...,1]))
    return loss

def style_loss(gt_dict={}, gen_dict={}):
    def loss(y_true, y_pred):
        total_variation_weight = 1.0
        style_weight = 1.0
        feature_layers = ['res_41', 'res_61',
                          'res_81', 'res_101',
                          'res_121']
        for layer_name in feature_layers:
            gen_features = gen_dict[layer_name]
            gt_features = gt_dict[layer_name]
            sl = style_loss(gt_features, gen_features)
            loss += (style_weight / len(feature_layers)) * sl
        loss += total_variation_weight * total_variation_loss(combination_image)
        return loss
    return loss

def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def gm_loss(gt, gen):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 2
    size = img_nrows * img_ncols
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