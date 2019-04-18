'''
Hyperparameters
'''
import re
import json

import numpy as np
import scipy.signal
import tensorflow as tf

# Hyperparameters are in CAPS
# TODO use tf.app.flags to parse hyperparams from input
#      or consider use json file to store hyperparams

class Hyperparameter:
    '''
    Contains hyperparameter settings
    '''
    pattern = r'[A-Z_]+'
    encoder_registry = {}
    estimator_registry = {}
    separator_registry = {}
    ozer_registry = {}
    dataset_registry = {}
    model_registry = {}

    def __init__(self):
        pass

    def digest(self):
        '''
        When hyperparameters are updated, this function should be called.

        This performs asserts, and derive some inferred hyperparams.
        '''
        self.COMPLEXX = dict(float32='complex64', float64='complex128')[self.FLOATX]
        self.FEATURE_SIZE = 1 + self.FFT_SIZE // 2
        assert isinstance(self.DROPOUT_KEEP_PROB, float)
        assert 0. < self.DROPOUT_KEEP_PROB <= 1.

        # FIXME: security concern by using eval?
        self.FFT_WND = eval(self.FFT_WND)

    def load(self, di):
        '''
        load from a dict

        Args:
            di: dict, string -> string
        '''
        assert isinstance(di, dict)
        pat = re.compile(self.pattern)
        for k,v in di.items():
            if None is pat.fullmatch(k):
                raise NameError
            assert isinstance(v, (str, int, float, bool, type(None)))
        self.__dict__.update(di)

    def load_json(self, file_):
        '''
        load from JSON file

        Args:
            file_: string or file-like
        '''
        if isinstance(file_, (str, bytes)):
            file_ = open(file_, 'r')
        di = json.load(file_)
        self.load(di)

    # decorators & getters
    @classmethod
    def register_encoder(cls_, name):
        def wrapper(cls):
            cls_.encoder_registry[name] = cls
            return cls
        return wrapper

    def get_encoder(self, name):
        return type(self).encoder_registry[name]

    @classmethod
    def register_estimator(cls_, name):
        def wrapper(cls):
            cls_.estimator_registry[name] = cls
            return cls
        return wrapper

    def get_estimator(self, name):
        return type(self).estimator_registry[name]

    @classmethod
    def register_separator(cls_, name):
        def wrapper(cls):
            cls_.separator_registry[name] = cls
            return cls
        return wrapper

    def get_separator(self, name):
        return type(self).separator_registry[name]

    @classmethod
    def register_optimizer(cls_, name):
        def wrapper(fn):
            cls_.ozer_registry[name] = fn
            return fn
        return wrapper

    def get_optimizer(self, name):
        return type(self).ozer_registry[name]

    @classmethod
    def register_dataset(cls_, name):
        def wrapper(fn):
            cls_.dataset_registry[name] = fn
            return fn
        return wrapper

    def get_dataset(self):
        return type(self).dataset_registry[self.DATASET_TYPE]

    @classmethod
    def register_model(cls_, name):
        def wrapper(fn):
            cls_.model_registry[name] = fn
            return fn
        return wrapper

    def get_model(self, name):
        return type(self).model_registry[name]

    def get_regularizer(self):
        reger = {
            None: (lambda _:None),
            'L1':tf.contrib.layers.l1_regularizer,
            'L2':tf.contrib.layers.l2_regularizer}[self.REG_TYPE](self.REG_SCALE)
        return reger


hparams = Hyperparameter()

