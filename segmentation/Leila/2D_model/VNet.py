import tensorflow as tf
from model.base_model import BaseModel
from model.ops import conv_2d, deconv_2d, Relu, max_pool
from utils import get_num_channels


class UNet(BaseModel):
    def __init__(self, sess, conf,
                 num_levels=4,
                 num_convs=(2, 2, 2, 2),
                 bottom_convs= 2,
                 act_fcn=Relu):

        super(UNet, self).__init__(sess, conf)
        # super().__init__(sess, conf)  Python3
        self.num_levels = num_levels
        self.num_convs = num_convs
        self.bottom_convs = bottom_convs
        self.k_size = self.conf.filter_size
        self.down_conv_factor = 2
        self.act_fcn = act_fcn
        self.build_network(self.x)
        self.configure_network()

    def build_network(self, x):
        # Building network...
        with tf.variable_scope('V-Net'):
            feature_list = list()

            with tf.variable_scope('Encoder'):
                for l in range(self.num_levels):
                    with tf.variable_scope('level_' + str(l + 1)):
                        x = self.conv_block_down(x, self.num_convs[l])
                        print('conv_{} shape: {}'.format(l+1, x.get_shape()))
                        feature_list.append(x)
                        if l==self.num_levels-1:
                            x = tf.nn.dropout(x, self.keep_prob)
                        x = self.down_conv(x)

            with tf.variable_scope('Bottom_level'):
                x = self.conv_block_down(x, self.bottom_convs)
                x = tf.nn.dropout(x, self.keep_prob)
                print('bottom_conv shape: {}'.format( x.get_shape()))

            with tf.variable_scope('Decoder'):
                for l in reversed(range(self.num_levels)):
                    with tf.variable_scope('level_' + str(l + 1)):
                        f = feature_list[l]
                        x = self.up_conv(x, tf.shape(f))
                        print('TU_{} shape: {}'.format(l + 1, x.get_shape()))
                        x = self.conv_block_up(x, f, self.num_convs[l])
                        print('conv_{} shape: {}'.format(l + 1, x.get_shape()))

            self.logits = conv_2d(x, 1, self.conf.num_cls, 'Output_layer', batch_norm=True,
                                  is_train=self.is_training)

    def conv_block_down(self, layer_input, num_convolutions):
        x = layer_input
        n_channels = get_num_channels(x)
        if n_channels == 1:
            n_channels = 64 #self.conf.start_channel_num
        else:
            n_channels = n_channels*2
        for i in range(num_convolutions):
            x = conv_2d(inputs=x,
                        filter_size=self.k_size,
                        num_filters=n_channels,
                        layer_name='conv_' + str(i + 1),
                        batch_norm=self.conf.use_BN,
                        is_train=self.is_training)
            #if i == num_convolutions - 1:
             #   x = x + layer_input
            x = self.act_fcn(x)
            #x = tf.nn.dropout(x, self.keep_prob)
        return x

    def conv_block_up(self, layer_input, fine_grained_features, num_convolutions):
        x = tf.concat((layer_input, fine_grained_features), axis=-1)
        print('concat shape: {}'.format(x.get_shape()))
        n_channels = get_num_channels(layer_input)
        print('channel#:{}'.format( n_channels))
        for i in range(num_convolutions):
            x = conv_2d(inputs=x,
                        filter_size=self.k_size,
                        num_filters=n_channels,
                        layer_name='conv_' + str(i + 1),
                        batch_norm=self.conf.use_BN,
                        is_train=self.is_training)
            #if i == num_convolutions - 1:
            #    x = x + layer_input
            x = self.act_fcn(x)
            #x = tf.nn.dropout(x, self.keep_prob)
        return x

    def down_conv(self, x):
        num_out_channels = get_num_channels(x) * 2
        #x = conv_2d(inputs=x,
         #           filter_size=2,
         #           num_filters=num_out_channels,
         #          layer_name='conv_down',
         #           stride=2,
         #           batch_norm=self.conf.use_BN,
         #           is_train=self.is_training,
         #           activation=self.act_fcn)
        x=max_pool(x, 2, 'max-pool')
        return x

    def up_conv(self, x, out_shape):
        num_out_channels = get_num_channels(x) // 2
        x = deconv_2d(inputs=x,
                      filter_size=2,
                      num_filters=num_out_channels,
                      layer_name='conv_up',
                      stride=2,
                      batch_norm=self.conf.use_BN,
                      is_train=self.is_training,
                      out_shape=out_shape,
                      activation=self.act_fcn)
        return x
