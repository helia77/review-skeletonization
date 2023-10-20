import tensorflow as tf
from model.base_model import BaseModel
from model.ops import conv_2d, deconv_2d, BN_Relu_conv_2d, max_pool
from utils import get_num_channels


class Tiramisu(BaseModel):
    def __init__(self, sess, conf,
                 num_levels=5,
                 num_convs=(4, 5, 7, 10, 12),
                 bottom_convs=15):

        super(Tiramisu, self).__init__(sess, conf)
        self.num_levels = num_levels
        self.num_convs = num_convs
        self.bottom_convs = bottom_convs
        self.k_size = self.conf.filter_size
        self.down_conv_factor = 2
        # BaseModel.__init__(self, sess, conf)

        # super().__init__(sess, conf)  Python3
        self.build_network(self.x)
        self.configure_network()

    def build_network(self, x):
        # Building network...
        with tf.variable_scope('Tiramisu'):
            feature_list = list()
            shape_list = list()

            with tf.variable_scope('input'):
                x = conv_2d(x, self.k_size, 48, 'input_layer', batch_norm=False, is_train=self.is_training)
                # x = tf.nn.dropout(x, self.keep_prob)
                print('{}: {}'.format('input_layer', x.get_shape()))

            with tf.variable_scope('Encoder'):
                for l in range(self.num_levels):
                    with tf.variable_scope('level_' + str(l + 1)):
                        level = self.dense_block(x, self.num_convs[l])
                        shape_list.append(tf.shape(level))
                        x = tf.concat((x, level), axis=-1)
                        print('{}: {}'.format('Encoder_level' + str(l + 1), x.get_shape()))
                        feature_list.append(x)
                        x = self.down_conv(x)

            with tf.variable_scope('Bottom_level'):
                x = self.dense_block(x, self.bottom_convs)
                print('{}: {}'.format('bottom_level', x.get_shape()))

            with tf.variable_scope('Decoder'):
                for l in reversed(range(self.num_levels)):
                    with tf.variable_scope('level_' + str(l + 1)):
                        f = feature_list[l]
                        shape_f = f.get_shape().as_list()
                        shape = x.get_shape().as_list()
                        #out_shape = [self.conf.batch_size] + list(map(lambda x: x * 2, shape[1:-1])) + [shape[-1]]
                        out_shape = [self.conf.batch_size] + shape_f[1:-1] + [shape[-1]]
                        # out_shape = tf.shape(tf.zeros(out_shape))
                        x = self.up_conv(x, out_shape=out_shape)
                        stack = tf.concat((x, feature_list[l]), axis=-1)
                        print('{}: {}'.format('Decoder_level' + str(l + 1), x.get_shape()))
                        x = self.dense_block(stack, self.num_convs[l])
                        print('{}: {}'.format('Dense_block_level' + str(l + 1), x.get_shape()))
                        stack = tf.concat((stack, x), axis=-1)
                        print('{}: {}'.format('stck_depth' + str(l + 1), stack.get_shape()))

            with tf.variable_scope('output'):

                print('{}: {}'.format('out_block_input', stack.get_shape()))
                self.logits = BN_Relu_conv_2d(stack, 1, self.conf.num_cls, 'Output_layer', batch_norm=True,
                                              is_train=self.is_training)
                print('{}: {}'.format('output', self.logits.get_shape()))

    def dense_block(self, layer_input, num_convolutions):
        x = layer_input
        layers = []
        # n_channels = get_num_channels(x)
        # if n_channels == self.conf.channel:
        #    n_channels = self.conf.start_channel_num
        for i in range(num_convolutions):
            layer = BN_Relu_conv_2d(inputs=x,
                                    filter_size=self.k_size,
                                    num_filters=self.conf.start_channel_num,
                                    layer_name='conv_' + str(i + 1),
                                    batch_norm=self.conf.use_BN,
                                    use_relu=True,
                                    is_train=self.is_training)
            layer = tf.nn.dropout(layer, self.keep_prob)
            layers.append(layer)
            x = tf.concat((x, layer), axis=-1)
        return tf.concat(layers, axis=-1)

    def down_conv(self, x):
        num_out_channels = get_num_channels(x)
        x = BN_Relu_conv_2d(inputs=x,
                            filter_size=1,
                            num_filters=num_out_channels,
                            layer_name='conv_down',
                            stride=1,
                            batch_norm=self.conf.use_BN,
                            is_train=self.is_training,
                            use_relu=True)
        x = tf.nn.dropout(x, self.keep_prob)
        x = max_pool(x, self.conf.pool_filter_size, name='maxpool')
        return x

    def up_conv(self, x, out_shape):
        num_out_channels = x.get_shape().as_list()[-1]
        x = deconv_2d(inputs=x,
                      filter_size=3,
                      num_filters=num_out_channels,
                      layer_name='conv_up',
                      stride=2,
                      batch_norm=False,
                      is_train=self.is_training,
                      out_shape=out_shape)
        return x
