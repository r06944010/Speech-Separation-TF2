from math import sqrt
from functools import partial

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from app.hparams import hparams
import os

class Model(object):
    def __init__(self, name='BaseModel'):
        self.name = name

    def save(self, save_path, step):
        model_name = self.name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not hasattr(self, 'saver'):
            self.saver = tf.train.Saver(self.model_vars)
        self.saver.save(self.sess,
                        os.path.join(save_path, model_name),
                        global_step=step)

    def load(self, save_path, model_file=None):
        print('[*] {} try loading ...'.format(self.name))
        self.model_vars = [var for var in tf.global_variables() if self.name in var.name]
        if not os.path.exists(save_path):
            print('[!] Checkpoints path does not exist, Loading model with fresh parameters')
            # self.sess.run(tf.initializers.variables(self.model_vars))
            return False

        print('[*] Reading checkpoints ...')
        if model_file is None:
            ckpt = tf.train.get_checkpoint_state(save_path)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            else:
                print('[!] Checkpoints path does not exist, Loading model with fresh parameters')
                # self.sess.run(tf.initializers.variables(self.model_vars))
                return False
        else:
            ckpt_name = model_file
        if not hasattr(self, 'saver'):
            self.saver = tf.train.Saver(self.model_vars)

        self.saver.restore(self.sess, os.path.join(save_path, ckpt_name))
        print('Read {}/{}'.format(save_path, ckpt_name))
        return True