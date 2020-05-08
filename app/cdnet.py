import itertools
import tensorflow as tf
from app.hparams import hparams
from app.modules import Model
import loss
import evaluation

import numpy as np
import scipy.signal

@hparams.register_model('cross-domain')
class CrossDomainModel(Model):
    def __init__(self, config, sess, name='cross-domain'):

        self.dtype = tf.float32
        self.config = config
        self.sess = sess
        self.name = name

        self.batch_size = config['training']['batch_size']
        self.n_speaker = config['training']['n_speaker']
        self.n_output = config['training']['n_output']
        self.n_anchor = config['training']['n_anchor']
        
        self.num_stacks = self.config['model']['num_stacks']
        self.dilations = [2 ** i for i in range(0, self.config['model']['dilations'])]
        self.input_length = self.output_length = config['model']['input_length']

        self.fft_len = config["model"]["kernel_size"]["ae"]
        self.fft_hop = config["model"]["hop"]

        self.stft_len = (self.input_length - (self.fft_len - self.fft_hop)) // self.fft_hop
        self.stft_ch = self.config["model"]["kernel_size"]["ae"]//2+1
        self.concat_channels = config["model"]["filters"]["ae"] + self.stft_ch

        self.output_ratio = config['model']['output_ratio']

        with tf.variable_scope(name):
            self.build()

    def fft_wnd(self, window_length, periodic=True, dtype=tf.float32, name=None):
        return np.sqrt(scipy.signal.hann(self.fft_len, sym=False).astype(np.float32))

    def build(self):
        self.audios = tf.placeholder(tf.float32, [self.batch_size, self.n_speaker, None], name='input_signals')
        self.mix_input = tf.reduce_sum(self.audios, axis=1)

        with tf.variable_scope("encoder"):
            # [batch, encode_len, channels]
            encoded_input = tf.layers.Conv1D(filters=self.config["model"]["filters"]["ae"],
                                             kernel_size=self.fft_len,
                                             strides=self.fft_hop,
                                             activation=tf.nn.relu,
                                             name="conv1d_relu")(tf.expand_dims(self.mix_input, -1))

        stfts_mix = tf.signal.stft(self.mix_input, frame_length=self.fft_len, frame_step=self.fft_hop, fft_length=self.fft_len,
                                   window_fn=self.fft_wnd)
        magni_mix = tf.abs(stfts_mix)
        phase_mix = tf.atan2(tf.imag(stfts_mix), tf.real(stfts_mix))

        with tf.variable_scope("bottle_start"):
            norm_input = self.cLN(tf.concat([encoded_input, tf.log1p(magni_mix)], axis=-1), "layer_norm")
            block_input = tf.layers.Conv1D(filters=self.config["model"]["filters"]["1*1-conv"], kernel_size=1)(norm_input)
                                                        
        for stack_i in range(self.num_stacks):
            for dilation in self.dilations:
                with tf.variable_scope("conv_block_{}_{}".format(stack_i, dilation)):
                    block_output = tf.layers.Conv1D(filters=self.config["model"]["filters"]["d-conv"],
                                                    kernel_size=1)(block_input)
                    block_output = self.prelu(block_output, name='1st-prelu', shared_axes=[1])
                    block_output = self.gLN(block_output, "first")
                    block_output = self._depthwise_conv1d(block_output, dilation)
                    block_output = self.prelu(block_output, name='2nd-prelu', shared_axes=[1])
                    block_output = self.gLN(block_output, "second")
                    block_output = tf.layers.Conv1D(filters=self.config["model"]["filters"]["1*1-conv"],
                                                    kernel_size=1)(block_output)
                    block_input += block_output

        if self.output_ratio == 1: 
            embed_channel = self.config["model"]["filters"]["ae"]
            feature_map = encoded_input
        elif self.output_ratio == 0: 
            embed_channel = self.stft_ch
            feature_map = magni_mix
        else:
            embed_channel = self.concat_channels
            feature_map = tf.concat([encoded_input, magni_mix], axis=-1)

        with tf.variable_scope('separator'):
            s_embed = tf.layers.Dense(embed_channel*self.config["model"]["embed_size"])(block_input)
            s_embed = tf.reshape(s_embed, [self.batch_size, -1, embed_channel, self.config["model"]["embed_size"]])

            # Estimate attractor from best combination from anchors
            v_anchors = tf.get_variable('anchors', [self.n_anchor, self.config["model"]["embed_size"]], 
                                        dtype=tf.float32)
            c_combs = tf.constant(list(itertools.combinations(range(self.n_anchor), self.n_speaker)), name='combs')
            s_anchor_sets = tf.gather(v_anchors, c_combs)

            s_anchor_assignment = tf.einsum('btfe,pce->bptfc', s_embed, s_anchor_sets)
            s_anchor_assignment = tf.nn.softmax(s_anchor_assignment)

            s_attractor_sets = tf.einsum('bptfc,btfe->bpce', s_anchor_assignment, s_embed)
            s_attractor_sets /= tf.expand_dims(tf.reduce_sum(s_anchor_assignment, axis=(2, 3)), -1)

            sp = tf.matmul(s_attractor_sets, tf.transpose(s_attractor_sets, [0, 1, 3, 2]))
            diag = tf.fill(sp.shape[:-1], float("-inf"))
            sp = tf.linalg.set_diag(sp, diag)

            s_in_set_similarities = tf.reduce_max(sp, axis=(-1, -2))

            s_subset_choice = tf.argmin(s_in_set_similarities, axis=1)
            s_subset_choice_nd = tf.transpose(tf.stack([tf.range(self.batch_size, dtype=tf.int64), s_subset_choice]))
            s_attractors = tf.gather_nd(s_attractor_sets, s_subset_choice_nd)

            s_logits = tf.einsum('btfe,bce->bctf', s_embed, s_attractors)
            output_code = s_logits * tf.expand_dims(feature_map, 1)

        with tf.variable_scope("decoder"):
            conv_out = pred_istfts = 0
            if self.output_ratio != 0: 
                output_frame = tf.layers.Dense(self.config["model"]["kernel_size"]["ae"])(output_code[...,:self.config["model"]["filters"]["ae"]])
                conv_out = tf.contrib.signal.overlap_and_add(signal=output_frame, frame_step=self.fft_hop)

            if self.output_ratio != 1: 
                phase_mix_expand = tf.expand_dims(phase_mix, 1)
                pred_stfts = tf.complex(tf.cos(phase_mix_expand)*output_code[...,-self.stft_ch:],
                                        tf.sin(phase_mix_expand)*output_code[...,-self.stft_ch:])
                pred_istfts = tf.signal.inverse_stft(pred_stfts, frame_length=self.fft_len, frame_step=self.fft_hop, fft_length=self.fft_len,
                                                     window_fn=tf.signal.inverse_stft_window_fn(self.fft_hop, forward_window_fn=self.fft_wnd))

            self.data_out = conv_out*self.output_ratio + pred_istfts*(1-self.output_ratio)


        self.loss, self.pred_output, self.sdr, perm = loss.pit_loss(
            self.audios, self.data_out, self.config, self.batch_size, self.n_speaker, self.n_output)


    def cLN(self, inputs, name):
        return tf.contrib.layers.layer_norm(
                    inputs=inputs,
                    begin_norm_axis=2,
                    begin_params_axis=-1)

    def gLN(self, inputs, name):
        return tf.contrib.layers.layer_norm(
                    inputs=inputs,
                    begin_norm_axis=1,
                    begin_params_axis=-1)

    def _depthwise_conv1d(self, inputs, di):
        inputs = tf.expand_dims(inputs, axis=1)
        filters = tf.get_variable(
            "dconv_filters",
            [1, self.config["model"]["kernel_size"]["d-conv"], self.config["model"]["filters"]["d-conv"], 1], 
            dtype=tf.float32)
        outputs = tf.nn.depthwise_conv2d(
            input=inputs,
            filter=filters,
            strides=[1, 1, 1, 1],
            padding='SAME',
            rate=[1, di])
        return tf.squeeze(outputs, axis=1)

    def prelu(self, _x, name, shared_axes=None, alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None):
        with tf.variable_scope(name):
            alphas = tf.get_variable('alpha', _x.shape[-1], dtype=tf.float32) # shape = _x.shape[-1]
            pos = tf.nn.relu(_x)
            neg = alphas * (_x - abs(_x)) * 0.5
            return pos + neg
