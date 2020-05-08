import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import itertools
import numpy as np

coeff = 10.0 / tf.log(10.0)
# K = 1e-10
K = 0.0

def snr(y_true, y_pred):
    noise = y_true - y_pred
    signal_pwr = tf.reduce_sum(y_true ** 2, -1)
    noise_pwr = tf.reduce_sum(noise ** 2, -1)
    return coeff * (tf.log(signal_pwr + K) - tf.log(noise_pwr + K))

def sdr(y_true, y_pred):
    s_target = dot(y_true, y_pred) * y_true / dot(y_true) 
    e_all = y_pred - s_target
    s_target_pwr = tf.reduce_sum(s_target ** 2, -1)
    e_all_pwr = tf.reduce_sum(e_all ** 2, -1)
    return coeff * (tf.log(s_target_pwr + K) - tf.log(e_all_pwr + K))
    
    # return coeff * (tf.log(_dot(y_true, y_pred)**2) - \
    #             tf.log(_dot(y_true,y_true) * _dot(y_pred,y_pred) - _dot(y_true, y_pred)**2))

    # return tf.squeeze(coeff * (tf.log(dot(y_true, y_pred)**2) - 
    #     tf.log(dot(y_true) * dot(y_pred) - dot(y_true, y_pred)**2)), -1)

def sdr_modify(y_true, y_pred):
    up = tf.reduce_sum(y_true * y_pred, -1)
    down = tf.sqrt(tf.reduce_sum(y_true ** 2, -1) * tf.reduce_sum(y_pred ** 2, -1))
    return up / down
    # Ven = tf.reduce_sum(y_true * y_pred, -1) ** 2 / tf.reduce_sum(y_pred ** 2, -1)
    # SDR = tf.sqrt(Ven / tf.reduce_sum(y_true ** 2, -1) )
    # return SDR

def sisnr(y_true, y_pred): # propose by conv-tasnet (which is identity to SDR ??)
    # tasnet : scale invariance is ensured by normalizing s_hat and s to zero-mean prior to the calculation
    y_true = y_true - tf.reduce_mean(y_true, -1, keepdims=True)
    y_pred = y_pred - tf.reduce_mean(y_pred, -1, keepdims=True)

    s_target = dot(y_true, y_pred) * y_true / dot(y_true)
    e_noise = y_pred - s_target
    s_target_pwr = tf.reduce_sum(s_target ** 2, -1)
    e_noise_pwr = tf.reduce_sum(e_noise ** 2, -1)
    return coeff * (tf.log(s_target_pwr + K) - tf.log(e_noise_pwr + K))

def dot(x,y=None):
    return tf.reduce_sum(x * y, -1, keepdims=True) if y != None \
        else tf.reduce_sum(x ** 2, -1, keepdims=True)

def _dot(x,y):
    return tf.reduce_sum(x*y, -1)