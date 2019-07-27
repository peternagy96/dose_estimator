import tensorflow as tf
from keras import backend as K


def lse(y_true, y_pred): # size: (batch, 5, 5, 1)
    loss = tf.reduce_mean(tf.squared_difference(y_pred, y_true))
    return loss

def mae(alpha=0.5): # size: (batch, 80, 80, 2)
    def loss(y_true, y_pred):
        return alpha * tf.reduce_mean(tf.abs(y_pred[...,0] - y_true[...,0])) + (1-alpha) * tf.reduce_mean(tf.abs(y_pred[...,1] - y_true[...,1]))
    return loss

def mae_g(alpha=0.5):
    def mae(y_true, y_pred):
        print(K.int_shape(y_true))
        return alpha*tf.reduce_mean(y_pred[...,0], y_true[...,0]) + (1-alpha)*tf.reduce_mean(y_pred[...,1], y_true[...,1])
    return lse


def cycle_loss(alpha=0.5): # size: (batch, 80, 80, 2)
    def loss(y_true, y_pred):
        return alpha * tf.reduce_mean(tf.abs(y_pred[...,0] - y_true[...,0])) + (1-alpha) * tf.reduce_mean(tf.abs(y_pred[...,1] - y_true[...,1]))
    return loss
