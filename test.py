import os
import tensorflow as tf
import numpy as np
from scipy import signal
import util
import separate
import json

from app.hparams import hparams


def test(config, cla):
    log_file = os.path.join(config['training']['path'], cla.ckpt_name, 'log_'+cla.test_set)

    if not os.path.exists(os.path.join(config['training']['path'], cla.ckpt_name)):
        os.mkdir(os.path.join(config['training']['path'], cla.ckpt_name))

    output_path = os.path.join(config['training']['path'], 'sample')

    sess = tf.Session()
    G = hparams.get_model(config['model']['type'])(config, sess)
    G_save_path = os.path.join(config['training']['path'], 'generat.ckpt')
    G.load(G_save_path, cla.ckpt_name)

    if not cla.mix_input_path.endswith('/'): cla.mix_input_path += '/'
    filenames = [filename for filename in os.listdir(cla.mix_input_path) if filename.endswith('.wav')]

    sdr_sum = []
    sisnr_sum = []
    pesq_sum = []

    for filename in filenames:
        util.myprint(log_file, filename)
        mix_audio = util.load_wav(cla.mix_input_path + filename, config['dataset']['sample_rate'])
        clean_1 = util.load_wav(cla.clean_input_path + 's1/' + filename, config['dataset']['sample_rate'])
        clean_2 = util.load_wav(cla.clean_input_path + 's2/' + filename, config['dataset']['sample_rate'])

        sdr, sisnr, pesq, pit_ch = separate.separate_sample(sess, G, config, mix_audio, clean_1, clean_2)

        util.myprint(log_file, '    sdr: {}, {}'.format(sdr[0], sdr[1]))
        util.myprint(log_file, '    sisnr: {}, {}'.format(sisnr[0], sisnr[1]))
        util.myprint(log_file, '    pesq: {}, {}'.format(pesq[0], pesq[1]))

        sdr_sum.append(sdr)
        sisnr_sum.append(sisnr)
        pesq_sum.append(pesq)

    sdr_sum = np.array(sdr_sum)
    sisnr_sum = np.array(sisnr_sum)
    pesq_sum = np.array(pesq_sum)

    util.myprint(log_file, 'test sdr : {}'.format(np.mean(sdr_sum)))
    util.myprint(log_file, 'test sisnr : {}'.format(np.mean(sisnr_sum)))
    util.myprint(log_file, 'test pesq : {}'.format(np.mean(pesq_sum)))
