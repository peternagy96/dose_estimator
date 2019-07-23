import tensorflow as tf


def lse_d(y_true, y_pred):
    loss = tf.reduce_mean(tf.squared_difference(y_pred, y_true))
    return loss

def lse_g(alpha=0.5):
    def lse(y_true, y_pred):
        return alpha*tf.reduce_mean(tf.squared_difference(y_pred[...,0], y_true[...,0])) + (1-alpha)*tf.reduce_mean(tf.squared_difference(y_pred[...,0], y_true[...,0]))
    return lse


def cycle_loss(y_true, y_pred):
    loss = tf.reduce_mean(tf.abs(y_pred - y_true))
    return loss
