import tensorflow as tf


def lse(self, y_true, y_pred):
    loss = tf.reduce_mean(tf.squared_difference(y_pred, y_true))
    return loss


def cycle_loss(self, y_true, y_pred):
    loss = tf.reduce_mean(tf.abs(y_pred - y_true))
    return loss
