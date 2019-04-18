# A Wavenet For Speech Denoising - Dario Rethage - 19.05.2017
# datasets.py

import util
import os
import numpy as np
import logging
from tqdm import tqdm
import tensorflow as tf

class WSJ0():

    def __init__(self, config, mode):
        self.mode = mode
        self.config = config
        self.spk = config['training']['n_speaker']
        self.sample_rate = config['dataset']['sample_rate']
        self.path = config['dataset']['path']
        self.batch_size = config['training']['batch_size']
        self.num_stacks = config['model']['num_stacks']
        self.dilations = [2 ** i for i in range(0, config['model']['dilations'] + 1)]

        self.input_length = config['model']['input_length']
        self.num_residual_blocks = len(self.dilations) * self.num_stacks

        if config['training']['path'] == "models/test":
            self.tfr = os.path.join(config['dataset']['path'], 'tfrecord', mode + '_' + 'debug.tfr')
        else:
            self.tfr = os.path.join(config['dataset']['path'], 'tfrecord',
                    str(config['training']['n_speaker']) + 'spk' + '_' + mode + '_' + str(self.input_length) + '.tfr')

        if not os.path.isfile(self.tfr) or config['dataset']['load_in_mem']:
            print('Find no {}'.format(self.tfr))
            print('[*] Load {} dataset in memory [*]'.format(self.mode))
            self.load_into_memory()

    
    def decode_dataset(self, serialized_example):
        example = tf.parse_single_example(
            serialized_example,
            features={
                "s1": tf.VarLenFeature(tf.float32),
                "s2": tf.VarLenFeature(tf.float32)
            },
        )
        s1 = tf.sparse_tensor_to_dense(example["s1"])
        s2 = tf.sparse_tensor_to_dense(example["s2"])
        audios = tf.stack([s1, s2])
        return audios

    def get_iterator(self):
        if os.path.isfile(self.tfr) and not self.config['dataset']['load_in_mem']:
            print("Loading data from \033[93m{} \033[0m".format(self.tfr))
            with tf.name_scope("input"):
                dataset = tf.data.TFRecordDataset(self.tfr).map(self.decode_dataset)
                if self.mode == "tr" or self.mode == "cv":
                    dataset = dataset.shuffle(self.batch_size * 100)
                dataset = dataset.batch(self.batch_size, drop_remainder=True)
                dataset = dataset.prefetch(self.batch_size * 10)
                self.iterator = dataset.make_initializable_iterator()
                return self.iterator.get_next()
        else:
            return None

    def load_into_memory(self):
        self.file_paths = {}
        self.sequences  = {'tr': {'a': [], 'b': []}, 'cv': {'a': [], 'b': []}, 'tt': {'a': [], 'b': []}}
        with open(os.path.join(self.path, 'merl_path', str(self.spk),'min', self.mode + '_1.txt'), 'r') as f:
            train_A = f.readlines()
            train_A = list(map(lambda _: _[:-1], train_A))
        with open(os.path.join(self.path, 'merl_path', str(self.spk),'min', self.mode + '_2.txt'), 'r') as f:
            train_B = f.readlines()
            train_B = list(map(lambda _: _[:-1], train_B))

        self.file_paths[self.mode] = {'a':train_A, 'b':train_B}
        self.file_base = [os.path.basename(f) for f in train_A]

        for spk in ['a', 'b']:
            sequences = self.load_directory(self.file_paths[self.mode][spk], spk)
            self.sequences[self.mode][spk] = sequences



    def load_directory(self, filenames, spk):
        sequences = []
        for filename in tqdm(filenames):
            sequence = util.load_wav(filename, self.sample_rate)
            sequences.append(sequence)
        sequences = np.array(sequences)

        return sequences

    def get_next(self):
        n_data = {'tr':20000,'cv':5000,'tt':3000}

        indices = np.arange((n_data[self.mode] + self.batch_size - 1) // self.batch_size * self.batch_size)
        indices %= n_data[self.mode]
        np.random.shuffle(indices)

        for i in range(len(indices)//self.batch_size):
            sample_indices = indices[i*self.batch_size:(i+1)*self.batch_size]
            batch_inputs = []

            for i, sample_i in enumerate(sample_indices):
                speech_a = self.sequences[self.mode]['a'][sample_i]
                speech_b = self.sequences[self.mode]['b'][sample_i]

                if len(speech_a) < self.input_length:
                    output_a = np.zeros((self.input_length))
                    output_a[:len(speech_a)] = speech_a
                    output_b = np.zeros((self.input_length))
                    output_b[:len(speech_b)] = speech_b
                else:
                    offset = np.squeeze(np.random.randint(0, len(speech_a) - self.input_length + 1, 1))
                    output_a = speech_a[offset:offset + self.input_length]
                    output_b = speech_b[offset:offset + self.input_length]

                batch_inputs.append([output_a, output_b])
            batch_inputs = np.array(batch_inputs, dtype='float32')
            yield batch_inputs, sample_indices

