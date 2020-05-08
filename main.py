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
import tensorflow as tf

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

def print_params(variables_names=None):
    if variables_names == None:
        variables_names = [v.name for v in tf.trainable_variables()]
    for k in variables_names:
        print(k)
    print('======================================================')

def debug():
    init = tf.global_variables_initializer()
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess.run(init)
    exit()

def check_init():
    uninitialized_vars = []
    for var in tf.global_variables():
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)
    for k in uninitialized_vars:
        print(k)
    init_new_vars_op = tf.initialize_variables(uninitialized_vars)

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

    for epoch in range(1, config['training']['num_epochs'] + 1):
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

                g_loss, g_sdr, g_curr_step, _ = sess.run(
                                        fetches=[G.loss, G.sdr, g_global_step, g_update],
                                        feed_dict={G.audios: feed_audio, g_learning_rate: glr})
                tr_loss += g_loss
                tr_sdr  += g_sdr
                tr_size += 1

                print('Train step {}: {} = {:5f}, sdr = {:5f}, lr = {}'.
                      format(g_curr_step, config['training']['loss'], g_loss, g_sdr, glr), end='\r')

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
