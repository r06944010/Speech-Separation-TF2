# A Wavenet For Speech Denoising - Dario Rethage - 19.05.2017
# Datasets.py

import util
import os
import numpy as np
import logging
from tqdm import tqdm
import tensorflow as tf

# from joblib import Parallel, delayed
# import multiprocessing

class prepro():
    def __init__(self, config, mode):
        self.mode = mode
        self.config = config
        self.sample_rate = config['dataset']['sample_rate']
        self.batch_size = config['training']['batch_size']
        self.num_stacks = config['model']['num_stacks']
        self.dilations = [2 ** i for i in range(0, config['model']['dilations'] + 1)]
        self.path = os.path.join(config['dataset']['path'], 
                        str(config['training']['n_speaker'])+"speakers/wav"+str(self.sample_rate//1000)+"k", 
                        config['dataset']['maxmin'])
        self.input_length = config['model']['input_length']

        self.tfr = os.path.join(config['dataset']['path'], 'tfrecord', 
                    str(config['training']['n_speaker']) + 'spk' + '_' + mode + '_' + str(self.input_length) + '.tfr')
        
        if not os.path.isfile(self.tfr):
            self.encode_dataset()
            print('Successfully save to {}'.format(self.tfr))
        else:
            print('{} exists!'.format(self.tfr))

    def encode_dataset(self):

        print('Encoding from {} into {}'.format(self.path, self.tfr))
        print('Input length : {}'.format(self.input_length))

        total = 0
        less_than_target = 0

        with tf.python_io.TFRecordWriter(self.tfr) as writer:      
            filenames = os.listdir(os.path.join(self.path, self.mode, 's1'))
            
            for filename in tqdm(filenames):
                s1 = util.load_wav(os.path.join(self.path, self.mode, 's1', filename), self.sample_rate)
                s2 = util.load_wav(os.path.join(self.path, self.mode, 's2', filename), self.sample_rate)

                def write(_s1, _s2):
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                "s1": tf.train.Feature(float_list=tf.train.FloatList(value=_s1)),
                                "s2": tf.train.Feature(float_list=tf.train.FloatList(value=_s2))
                            }))
                    writer.write(example.SerializeToString())
                    
                if len(s1) < self.input_length:
                    # b = np.random.random_integers(self.receptive_field_length//2, 
                    #                               self.input_length-len(s1)-self.receptive_field_length//2)
                    b = np.random.random_integers(0, self.input_length-len(s1))
                    s1_pad = np.zeros((self.input_length))
                    s1_pad[b:b+len(s1)] = s1
                    s2_pad = np.zeros((self.input_length))
                    s2_pad[b:b+len(s2)] = s2
                    write(s1_pad, s2_pad)
                    less_than_target += 1

                else:
                    stride = self.input_length // 2
                    for i in range(0, len(s1) - self.input_length, stride):
                        s1_pad = s1[i:i+self.input_length]
                        s2_pad = s2[i:i+self.input_length]
                        write(s1_pad, s2_pad)
                        total += 1
                        
            print('total example : {}, less than target : {}'.format(total + less_than_target, less_than_target))
    

