# A Wavenet For Speech Denoising - Dario Rethage - 19.05.2017
# Denoise.py

from __future__ import division
import os
import util
import tqdm
import numpy as np
import scipy.signal
import mir_eval
import itertools
from pypesq import pesq

def signal_to_distortion_ratio(x,y):
    def _dot(x, y):
        return np.sum(np.multiply(x ,y), -1)
    return 10 * np.log10(np.square(_dot(x,y)) / (_dot(x,x)*_dot(y,y) - np.square(_dot(x,y))))

def separate_sample(sess, model, config, mix, c1, c2):

    batch_size = config['training']['batch_size']
    n_channel  = config['training']['n_output']
    n_speaker  = config['training']['n_speaker']

    stride = config['model']['hop']

    num_output_samples = mix.shape[0]
    num_fragments = int(np.ceil(num_output_samples / model.input_length))
    target_field_length = model.input_length
    target_padding = 0

    num_batches = int(np.ceil(num_fragments / batch_size))

    output = [[] for _ in range(n_channel)]
    num_pad_values = 0
    fragment_i = 0

   # pad input mixture to 10x, since stride is 10
    num_pad_values = stride - mix.shape[0] % stride
    mix = np.pad(mix, (0, num_pad_values), mode='constant', constant_values=0)

    output, = sess.run(fetches=[model.data_out], feed_dict={model.mix_input: np.expand_dims(mix, 0)})[0]

    output = np.array(output)
    if num_pad_values != 0:
        output = output[:,:-num_pad_values]
        mix = mix[:-num_pad_values]

    clean_wav = np.array([c1, c2])

    perms = np.array(list(itertools.permutations(range(n_channel), n_speaker)))
    perms_onehot = (np.arange(perms.max()+1) == perms[...,None]).astype(int)

    cross_sdr = signal_to_distortion_ratio(np.expand_dims(np.array([c1,c2]), 1), 
                                           np.expand_dims(output, 0))
    loss_sets = np.einsum('ij,pij->p', cross_sdr, perms_onehot)
    best_perm = perms[np.argmax(loss_sets)]
    pit_output = output[best_perm]

    # SDR
    _sdr, _sir, _sar, _perm = mir_eval.separation.bss_eval_sources(clean_wav, pit_output)
    # SISNR
    clean_wav_norm = clean_wav - np.mean(clean_wav, axis=-1, keepdims=True)
    pit_output_norm = pit_output - np.mean(pit_output, axis=-1, keepdims=True)
    _sisnr = signal_to_distortion_ratio(clean_wav_norm, pit_output_norm)
    # PESQ
    _pesq = [pesq(clean_wav[0], pit_output[0], 8000), pesq(clean_wav[1], pit_output[1], 8000)]

    perm_output = np.expand_dims(best_perm, -1).tolist()

    return _sdr, _sisnr, _pesq, perm_output