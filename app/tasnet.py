import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from app.hparams import hparams
from app.modules import Model
import loss
import evaluation

@hparams.register_model('tasnet')
class Tasnet(Model):
    def __init__(self, config, sess, name='tasnet'):

        self.dtype = tf.float32
        self.config = config
        self.sess = sess
        self.name = name
        self.batch_size = config['training']['batch_size']
        self.n_speaker = config['training']['n_speaker'] if 'n_speaker' in config['training'] else 2
        self.n_output = config['training']['n_output']  if 'n_output' in config['training'] else 2
        self.mute_other_channel = config['training']['mute_other_channel'] if 'mute_other_channel' in config['training'] else False
        self.skip_connect = config['model']['skip'] if 'skip' in config['model'] else False
        self.use_sigmoid = config['model']['sigmoid'] if 'sigmoid' in config['model'] else False

        self.num_stacks = self.config['model']['num_stacks']
        self.dilations = [2 ** i for i in range(0, self.config['model']['dilations'])]
        self.input_length = self.output_length = config['model']['input_length']
        
        with tf.variable_scope(name):
            self.build_tasnet()

    def build_tasnet(self):
        # [self.batch_size, self.n_speaker, self.input_length]
        self.audios = tf.placeholder(tf.float32, [self.batch_size, self.n_speaker, None], name='input_signals')
        self.mix_input = tf.reduce_sum(self.audios, axis=1) # [self.batch_size, self.input_length]

        with tf.variable_scope("encoder"):
            # [batch, encode_len, channels]
            encoded_input = tf.layers.Conv1D(filters=self.config["model"]["filters"]["ae"],
                                             kernel_size=self.config["model"]["kernel_size"]["ae"],
                                             strides=self.config["model"]["kernel_size"]["ae"]//2,
                                             activation=tf.nn.relu,
                                             name="conv1d_relu")(tf.expand_dims(self.mix_input, -1))

        with tf.variable_scope("bottle_start"):
            norm_input = self.cLN(encoded_input, "layer_norm")
            block_input = tf.layers.Conv1D(filters=self.config["model"]["filters"]["1*1-conv"],
                                                 kernel_size=1)(norm_input)
        skip_connections = []
        for stack_i in range(self.num_stacks):
            for dilation in self.dilations:
                block_name = "conv_block_{}_{}".format(stack_i, dilation)
                with tf.variable_scope(block_name):
                    block_output = tf.layers.Conv1D(filters=self.config["model"]["filters"]["d-conv"],
                                                    kernel_size=1)(block_input)
                    block_output = self.prelu(block_output, name='1st-prelu', shared_axes=[1])
                    block_output = self.gLN(block_output, "first")
                    block_output = self._depthwise_conv1d(block_output, dilation)
                    block_output = self.prelu(block_output, name='2nd-prelu', shared_axes=[1])
                    block_output = self.gLN(block_output, "second")

                    if self.skip_connect:
                        block_skip = tf.layers.Conv1D(filters=self.config["model"]["filters"]["sc"],
                                                      kernel_size=1)(block_output)
                        skip_connections.append(block_skip)


                    block_output = tf.layers.Conv1D(filters=self.config["model"]["filters"]["1*1-conv"],
                                                    kernel_size=1)(block_output)
                    block_input += block_output
                    

        with tf.variable_scope("bottle_end"):
            if self.skip_connect:
                block_input = tf.reduce_sum(skip_connections, axis=0)
                block_input = self.prelu(block_input, name='skip-prelu', shared_axes=[1])
            
            mask_list = [tf.layers.Conv1D(filters=self.config["model"]["filters"]["ae"], 
                                                kernel_size=1,
                                                name="1x1_conv_decoder_{}".format(i))(block_input)
                         for i in range(self.n_output)]
            
            if self.use_sigmoid:
                probs = tf.nn.sigmoid(tf.stack(mask_list, axis=1)) # [batch, encoded_len, ae_channel, # output]
            else:
                probs = tf.stack(mask_list, axis=1) # [batch, encoded_len, ae_channel, # output]

            output_code = probs * tf.expand_dims(encoded_input, axis=1)

        with tf.variable_scope("decoder"):
            # [batch, # output, encoded_len, ae_channel]
            output_frame = tf.layers.Dense(self.config["model"]["kernel_size"]["ae"])(output_code)
            # [batch, # output, encoded_len, ae_kernel_size]
            self.data_out = tf.signal.overlap_and_add(
                              signal=output_frame, frame_step=self.config["model"]["kernel_size"]["ae"]//2)


        # for pit at training, or valid and test => dynamic label assignment
        self.loss, self.pred_output, self.sdr, self.perm_idxs = loss.pit_loss(
            self.audios, self.data_out, self.config, self.batch_size, self.n_speaker, self.n_output)
        
        # for fixed assignment
        self.fixed_perm = tf.placeholder(tf.int32, [None, self.n_speaker])
        self.loss_fix, self.pred_output_fix, self.sdr_fix, self.perm_idxs_fix = loss.fixed_loss(
            self.audios, self.data_out, self.config, self.batch_size, self.n_speaker, self.n_output,
            s_perm_choose=self.fixed_perm)            


    def cLN(self, inputs, name):
        # inputs: [batch_size, some len, channel_size]

        with tf.variable_scope('LayerNorm'):
            channel_size = inputs.shape[-1]
            E = tf.reduce_mean(inputs, axis=[2], keepdims=True)
            Var = tf.reduce_mean((inputs - E)**2, axis=[2], keepdims=True)
            gamma = tf.Variable(tf.ones(shape=(channel_size)), dtype=self.dtype, name='gamma')
            gamma = tf.reshape(gamma, [1,1,-1])
            beta = tf.Variable(tf.zeros(shape=(channel_size)), dtype=self.dtype, name='beta')
            beta = tf.reshape(beta, [1,1,-1])
            return ((inputs - E) / (Var + 1e-8)**0.5) * gamma + beta


    def gLN(self, inputs, name):
        # inputs: [batch_size, some len, channel_size]

        with tf.variable_scope('LayerNorm'):
            channel_size = inputs.shape[-1]
            E = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
            Var = tf.reduce_mean((inputs - E)**2, axis=[1, 2], keepdims=True)
            gamma = tf.Variable(tf.ones(shape=(channel_size)), dtype=self.dtype, name='gamma')
            gamma = tf.reshape(gamma, [1,1,-1])
            beta = tf.Variable(tf.zeros(shape=(channel_size)), dtype=self.dtype, name='beta')
            beta = tf.reshape(beta, [1,1,-1])
            return ((inputs - E) / (Var + 1e-8)**0.5) * gamma + beta
        

    def _depthwise_conv1d(self, inputs, di):
        # inputs  : NHWC
        # filters : [filter_height, filter_width, in_channels, channel_multiplier]
        # rate    : [height, width]
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
        # f(x) = alpha * x for x < 0, f(x) = x for x >= 0
        with tf.variable_scope(name):
            alphas = tf.get_variable('alpha', _x.shape[-1], dtype=tf.float32) # shape = _x.shape[-1]
            pos = tf.nn.relu(_x)
            neg = alphas * (_x - abs(_x)) * 0.5
            return pos + neg
