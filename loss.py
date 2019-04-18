import numpy as np
import tensorflow as tf
import itertools
import evaluation

def pit_loss(y_true, y_pred, config, batch_size, n_speaker, n_output, pit_axis=1):
    # [batch, spk #, length]
    real_spk_num = n_speaker

    # TODO 1: # output channel != # speaker
    loss_type = config['training']['loss']

    v_perms = tf.constant(list(itertools.permutations(range(n_output), n_speaker)))
    v_perms_onehot = tf.one_hot(v_perms, n_output)

    y_true_exp = tf.expand_dims(y_true, pit_axis+1) # [batch, n_speaker, 1,        len]
    y_pred_exp = tf.expand_dims(y_pred, pit_axis)   # [batch, 1,         n_output, len]

    if loss_type == 'l1':
        y_cross_loss = y_true_exp - y_pred_exp
        y_cross_loss_abs = tf.reduce_sum(tf.abs(y_cross_loss), axis=3)
        cross_total_loss = y_cross_loss_abs

    elif loss_type == 'l2':
        y_cross_loss = y_true_exp - y_pred_exp
        y_cross_loss_abs = tf.reduce_sum(tf.square(y_cross_loss), axis=3)
        cross_total_loss = y_cross_loss_abs

    elif loss_type == 'snr':
        cross_total_loss = -evaluation.snr(y_true_exp, y_pred_exp)

    elif loss_type == 'sdr':
        cross_total_loss = -evaluation.sdr(y_true_exp, y_pred_exp)

    elif loss_type == 'sisnr':
        cross_total_loss = -evaluation.sisnr(y_true_exp, y_pred_exp)

    elif loss_type == 'sdr_modify':
        cross_total_loss = -evaluation.sdr_modify(y_true_exp, y_pred_exp)

    loss_sets = tf.einsum('bij,pij->bp', cross_total_loss, v_perms_onehot) 
    loss = tf.reduce_min(loss_sets, axis=1)
    loss = tf.reduce_mean(loss)
        
    # find permutation sets for y pred
    s_perm_sets = tf.argmin(loss_sets, 1)
    s_perm_choose = tf.gather(v_perms, s_perm_sets)
    s_perm_idxs = tf.stack([
        tf.tile(
            tf.expand_dims(tf.range(batch_size), 1),
            [1, n_speaker]),
        s_perm_choose], axis=2)

    s_perm_idxs = tf.reshape(s_perm_idxs, [batch_size*n_speaker, 2])
    y_pred = tf.gather_nd(y_pred, s_perm_idxs)
    y_pred = tf.reshape(y_pred, [batch_size, n_speaker, -1])

    if loss_type != 'sdr':
        sdr = evaluation.sdr(y_true[:,:real_spk_num,:], y_pred[:,:real_spk_num,:])
        sdr = tf.reduce_mean(sdr)
    else:
        sdr = -loss/2

    # if config['model']['discriminator']:
    #     y_true_mix = tf.reduce_sum(y_true, 1)
    #     y_pred_mix = tf.reduce_sum(y_pred, 1)
    #     loss = -tf.reduce_mean(tf.abs(y_true_mix - y_pred_mix))

    return loss, y_pred, sdr, s_perm_choose
