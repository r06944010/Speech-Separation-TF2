# python function
import sys
import logging
import optparse
import argparse
import json
import os
import csv

# my function
import datasets
import util
import separate
from app.hparams import hparams
import test

import numpy as np
import scipy.signal
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from app.hparams import hparams
from preprocess import prepro
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def get_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', default='train')
    parser.add_argument('--config', '-c', default='json/config.json')
    parser.add_argument('--ckpt_name', '-ckpt')
    parser.add_argument('--mix_input_path', '-mix', default='./wsj0-mix/2speakers/wav8k/min/tt/mix/')
    parser.add_argument('--clean_input_path',  '-clean', default='./wsj0-mix/2speakers/wav8k/min/tt/')
    parser.add_argument('--test_set',  '-ts', default='tt')
    options = parser.parse_args()
    return options


def load_config(config_filepath):
    try:
        config_file = open(config_filepath, 'r')
    except IOError:
        logging.error('No readable config file at path: ' + config_filepath)
        exit()
    else:
        with config_file:
            return json.load(config_file)

def get_dataset(config, mode):
    if config['dataset']['type'] == 'wsj0-mix':
        return datasets.WSJ0(config, mode)
    else:
        print('Dataset not Implemented : ' + config['dataset']['type'])
        exit()

def training(config, cla):
    g_global_step = tf.Variable(0, trainable=False, name=config['model']['type']+"_global_step")

    glr = config['optimizer']['lr']
    sess = tf.Session()

    # build model
    G = hparams.get_model(config['model']['type'])(config, sess)

    ## update params
    G_vars = [var for var in tf.trainable_variables() if config['model']['type'] in var.name]
    util.count_params(G_vars, config['model']['type'])
    util.total_params()

    g_learning_rate = tf.placeholder(tf.float32, [])

    g_ozer = hparams.get_optimizer(config['optimizer']['type'])(learn_rate=g_learning_rate)
    
    g_grad = g_ozer.compute_gradients(G.loss, G_vars)
    g_update = g_ozer.apply_gradients(g_grad, global_step=g_global_step)

    g_grad_fix = g_ozer.compute_gradients(G.loss_fix, G_vars)
    g_update_fix = g_ozer.apply_gradients(g_grad_fix, global_step=g_global_step)

    ## restore from checkpoint
    G_save_path = os.path.join(config['training']['path'], 'generat.ckpt')

    sess.run(tf.global_variables_initializer())

    G.load(G_save_path)

    history_file = os.path.join(config['training']['path'], 'history.txt')

    tr_dataset = get_dataset(config, 'tr')
    cv_dataset = get_dataset(config, 'cv')
    tr_next = tr_dataset.get_iterator()
    cv_next = cv_dataset.get_iterator()

    valid_best_sdr = float('-inf')
    valid_wait = 0

    if config['training']['perm_path'] != None:
        fixed_perm_list = util.read_pretrained_perm(config['training']['perm_path'], tr_dataset.file_base)

    last_step = sess.run(g_global_step)
    tr_audio_perm = {i:[] for i in range(20000)} if last_step == 0 else io_tool.load_perm(config, 'tr', last_step, tr_dataset, 20000)
    cv_audio_perm = {i:[] for i in range(5000)} if last_step == 0 else io_tool.load_perm(config, 'cv', last_step, cv_dataset, 5000)

    for epoch in range(last_step//(20000//config['training']['batch_size'])+1, config['training']['num_epochs'] + 1):
        
        tr_loss = tr_size = tr_sdr = 0.0
        
        util.myprint(history_file, '-' * 20 + ' epoch {} '.format(epoch) + '-' * 20)

        ## training data initial
        if hasattr(tr_dataset, 'iterator'):
            sess.run(tr_dataset.iterator.initializer)
        else:
            tr_gen = tr_dataset.get_next()

        while True:
            try:
                feed_audio, audio_idx = sess.run(tr_next) if tr_next != None else next(tr_gen)

                if config['training']['pit'] == True:
                    g_loss, g_sdr, g_curr_step, _, g_perm_idx = sess.run(
                                            fetches=[G.loss, G.sdr, g_global_step, g_update, G.perm_idxs],
                                            feed_dict={G.audios: feed_audio, g_learning_rate: glr})

                elif config['training']['perm_path'] != None:
                    fixed_perm = np.take(fixed_perm_list, audio_idx, axis=0)
                    g_loss, g_sdr, g_curr_step, _, g_perm_idx = sess.run(
                                            fetches=[G.loss_fix, G.sdr_fix, g_global_step, g_update_fix, G.perm_idxs_fix],
                                            feed_dict={G.audios: feed_audio, g_learning_rate: glr, G.fixed_perm: fixed_perm})

                tr_loss += g_loss
                tr_sdr  += g_sdr
                tr_size += 1

                print('Train step {}: {} = {:5f}, sdr = {:5f}, lr = {}'.
                      format(g_curr_step, config['training']['loss'], g_loss, g_sdr, glr), end='\r')

                # record label assignment
                for _i, _id in enumerate(audio_idx):
                    tr_audio_perm[_id].append(g_perm_idx[_i].tolist())

            except (tf.errors.OutOfRangeError, StopIteration):
                util.myprint(history_file, 'Train step {}: {} = {:5f}, sdr = {:5f}, lr = {}'.
                            format(g_curr_step, config['training']['loss'], g_loss, g_sdr, glr))
                util.myprint(history_file, 'mean {} = {:5f} , mean sdr = {:5f}, lr = {}'.
                            format(config['training']['loss'], tr_loss/tr_size, tr_sdr/tr_size, glr))
                break

        ## valid iteration
        if hasattr(cv_dataset, 'iterator'):
            sess.run(cv_dataset.iterator.initializer)
        else:
            cv_gen = cv_dataset.get_next()

        cv_loss = cv_size = cv_sdr = 0.0
        while True:
            try:
                feed_audio, audio_idx = sess.run(cv_next) if cv_next != None else next(cv_gen)
                g_loss, g_sdr = sess.run(fetches=[G.loss, G.sdr], feed_dict={G.audios: feed_audio})

                cv_loss += g_loss
                cv_sdr  += g_sdr
                cv_size += 1

            except (tf.errors.OutOfRangeError, StopIteration):
                curr_loss = cv_loss/cv_size
                curr_sdr = cv_sdr/cv_size
                util.myprint(history_file, 'Valid '+ config['training']['loss'] +' = {:5f}, sdr = {}'.\
                                format(curr_loss, curr_sdr))
                
                ## save model for every improve of the best valid score
                ## or last epoch
                if curr_sdr > valid_best_sdr or epoch == config['training']['num_epochs']:
                    util.myprint(history_file, 'Save Model')
                    valid_wait = 0
                    valid_best_sdr = curr_sdr
                    G.save(G_save_path, g_curr_step)

                else:
                    valid_wait += 1
                    if valid_wait == config['training']['half_lr_patience']:
                        glr /= 2; valid_wait = 0
                break

        util.write(os.path.join(config['training']['path'], 'tr_perm.csv'), tr_dataset.file_base, tr_audio_perm, epoch, config['training']['n_speaker'])

if __name__ == "__main__":

    cla = get_command_line_arguments()
    config = load_config(cla.config)
    print('Save model path : {}'.format(config['training']['path']))

    if not os.path.exists(config['training']['path']):
        os.mkdir(config['training']['path'])

    if cla.mode == 'train':
        util.pretty_json_dump(config, os.path.join(config['training']['path'], os.path.basename(cla.config)))
        training(config, cla)

    elif cla.mode == 'test':
        test.test(config, cla)
