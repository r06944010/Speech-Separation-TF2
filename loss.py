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

    cross_total_loss = get_loss(loss_type, y_true_exp, y_pred_exp)

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
        sdr = -loss/n_speaker

    return loss, y_pred, sdr, s_perm_choose


def fixed_loss(t_true, t_pred, config, batch_size, n_speaker, n_output, s_perm_choose=None):
    # use pretrained permutation
    s_perm_idxs = tf.stack([
        tf.tile(
            tf.expand_dims(tf.range(batch_size), 1),
            [1, n_speaker]),
        s_perm_choose], axis=2)

    s_perm_idxs = tf.reshape(s_perm_idxs, [batch_size*n_speaker, 2])
    t_pred = tf.gather_nd(t_pred, s_perm_idxs)
    t_pred = tf.reshape(t_pred, [batch_size, n_speaker, -1])

    loss_type = config['training']['loss']
    loss = get_loss(loss_type, t_true, t_pred)
    loss = tf.reduce_mean(loss)

    if loss_type != 'sdr':
        sdr = evaluation.sdr(t_true, t_pred)
        sdr = tf.reduce_mean(sdr)
    else:
        sdr = -loss/n_speaker

    return loss, t_pred, sdr, s_perm_choose

def get_loss(loss_type, t_true_exp, t_pred_exp, axis=-1):
    if loss_type == 'l1':
        y_cross_loss = t_true_exp - t_pred_exp
        cross_total_loss = tf.reduce_sum(tf.abs(y_cross_loss), axis=axis)

    elif loss_type == 'l2':
        y_cross_loss = t_true_exp - t_pred_exp
        cross_total_loss = tf.reduce_sum(tf.square(y_cross_loss), axis=axis)

    elif loss_type == 'snr':
        cross_total_loss = -evaluation.snr(t_true_exp, t_pred_exp)

    elif loss_type == 'sdr':
        cross_total_loss = -evaluation.sdr(t_true_exp, t_pred_exp)

    elif loss_type == 'sisnr':
        cross_total_loss = -evaluation.sisnr(t_true_exp, t_pred_exp)

    elif loss_type == 'sdr_modify':
        cross_total_loss = -evaluation.sdr_modify(t_true_exp, t_pred_exp)

    elif loss_type == 'sisdr':
        cross_total_loss = -evaluation.sisdr(t_true_exp, t_pred_exp)

    elif loss_type == 'sym_sisdr':
        cross_total_loss = -evaluation.sym_sisdr(t_true_exp, t_pred_exp)

    return cross_total_loss