import tensorflow as tf

from utils import get_num_channels


def weight_variable(name, shape):
    """
    Create a weight variable with appropriate initialization
    :param name: weight name
    :param shape: weight shape
    :return: initialized weight variable
    """
    initer = tf.contrib.layers.xavier_initializer(uniform=False)
    return tf.get_variable('W_' + name, dtype=tf.float32,
                           shape=shape, initializer=initer)


def bias_variable(name, shape):
    """
    Create a bias variable with appropriate initialization
    :param name: bias variable name
    :param shape: bias variable shape
    :return: initial bias variable
    """
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b_' + name, dtype=tf.float32,
                           initializer=initial)


def conv_2d(inputs, filter_size, num_filters, layer_name, stride=1, is_train=True,
            batch_norm=False, add_reg=False, activation=tf.identity):
    """
    Create a 2D convolution layer
    :param inputs: input array
    :param filter_size: size of the filter
    :param num_filters: number of filters (or output feature maps)
    :param layer_name: layer name
    :param stride: convolution filter stride
    :param batch_norm: boolean to use batch norm (or not)
    :param is_train: boolean to differentiate train and test (useful when applying batch normalization)
    :param add_reg: boolean to add norm-2 regularization (or not)
    :param activation: type of activation to be applied
    :return: The output array
    """
    num_in_channel = get_num_channels(inputs)
    with tf.variable_scope(layer_name):
        shape = [filter_size, filter_size, num_in_channel, num_filters]
        weights = weight_variable(layer_name, shape=shape)
        tf.summary.histogram('W', weights)
        layer = tf.nn.conv2d(input=inputs,
                             filter=weights,
                             strides=[1, stride, stride, 1],
                             padding="SAME")
        #print('{}: {}'.format(layer_name, layer.get_shape()))
        if batch_norm:
            layer = batch_norm_wrapper(layer, is_train)
        else:
            biases = bias_variable(layer_name, [num_filters])
            layer += biases
        layer = activation(layer)
        if add_reg:
            tf.add_to_collection('reg_weights', weights)
    return layer


def deconv_2d(inputs, filter_size, num_filters, layer_name, stride=1, batch_norm=False,
              is_train=True, add_reg=False, activation=tf.identity, out_shape=None):
    """
    Create a 2D transposed-convolution layer
    :param inputs: input array
    :param filter_size: size of the filter
    :param num_filters: number of filters (or output feature maps)
    :param layer_name: layer name
    :param stride: convolution filter stride
    :param batch_norm: boolean to use batch norm (or not)
    :param is_train: boolean to differentiate train and test (useful when applying batch normalization)
    :param add_reg: boolean to add norm-2 regularization (or not)
    :param activation: type of activation to be applied
    :param out_shape: Tensor of output shape
    :return: The output array
    """
    input_shape = inputs.get_shape().as_list()
    with tf.variable_scope(layer_name):
       # kernel_shape = [filter_size, filter_size, num_filters, input_shape[-1]]
        #if not len(out_shape.get_shape().as_list()):    # if out_shape is not provided
        #    out_shape = [input_shape[0]] + list(map(lambda x: x*2, input_shape[1:-1])) + [num_filters]
        #weights = weight_variable(layer_name, shape=kernel_shape)
        # biases = bias_variable(layer_name, [num_filters])
        # layer = tf.nn.conv2d_transpose(inputs,
        #                                filter=weights,
        #                                output_shape=out_shape,
        #                                strides=[1, stride, stride, 1],
        #
        #           padding="SAME")
        layer = tf.layers.conv2d_transpose(inputs,
                                       filters=num_filters,
                                       kernel_size=[filter_size,filter_size],
                                       strides=[stride, stride],
                                       padding="SAME")
        #print('{}: {}'.format(layer_name, layer.get_shape()))
        if batch_norm:
            layer = batch_norm_wrapper(layer, is_train)
        else:
            biases = bias_variable(layer_name, [num_filters])
            layer += biases
        layer = activation(layer)
        #if add_reg:
          #  tf.add_to_collection('weights', weights)
    return layer


def BN_Relu_conv_2d(inputs, filter_size, num_filters, layer_name, stride=1, is_train=True,
                    batch_norm=True, use_relu=True, add_reg=False):
    """
    Create a BN, ReLU, and 2D convolution layer
    :param inputs: input array
    :param filter_size: size of the filter
    :param num_filters: number of filters (or output feature maps)
    :param layer_name: layer name
    :param stride: convolution filter stride
    :param batch_norm: boolean to use batch norm (or not)
    :param is_train: boolean to differentiate train and test (useful when applying batch normalization)
    :param add_reg: boolean to add norm-2 regularization (or not)
    :param use_relu:
    :return: The output array
    """
    num_in_channel = get_num_channels(inputs)
    with tf.variable_scope(layer_name):
        if batch_norm:
            inputs = batch_norm_wrapper(inputs, is_train)
        if use_relu:
            inputs = tf.nn.relu(inputs)
        shape = [filter_size, filter_size, num_in_channel, num_filters]
        weights = weight_variable(layer_name, shape=shape)
        layer = tf.nn.conv2d(input=inputs,
                             filter=weights,
                             strides=[1, stride, stride, 1],
                             padding="SAME")
        if add_reg:
            tf.add_to_collection('reg_weights', weights)
    return layer


def max_pool(x, ksize, name):
    """
    Create a 2D max-pooling layer
    :param x: input to max-pooling layer
    :param ksize: size of the max-pooling filter
    :param name: layer name
    :return: The output array
    """
    with tf.variable_scope(name):
        maxpool = tf.nn.max_pool(x,
                                   ksize=[1, ksize, ksize, 1],
                                   strides=[1, 2, 2, 1],
                                   padding="SAME",
                                   name=name)
        print('{}: {}'.format(name, maxpool.get_shape()))
        return maxpool


def batch_norm_wrapper(inputs, is_training, decay=0.999, epsilon=1e-3):
    """
    creates a batch normalization layer
    :param inputs: input array
    :param is_training: boolean for differentiating train and test
    :param decay:
    :param epsilon:
    :return: normalized input
    """
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        if len(inputs.get_shape().as_list()) == 4:  # For 2D convolutional layers
            batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
        else:  # For fully-connected layers
            batch_mean, batch_var = tf.nn.moments(inputs, [0])
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)

def avg_pool(x, ksize, stride, scope):
    """Create an average pooling layer."""
    return tf.nn.avg_pool(x,
                            ksize=[1, ksize, ksize, 1],
                            strides=[1, stride, stride, 1],
                            padding="VALID",
                            name=scope)

def batch_norm(inputs, is_training, scope='BN', decay=0.999, epsilon=1e-3):
    """
    creates a batch normalization layer
    :param inputs: input array
    :param is_training: boolean for differentiating train and test
    :param scope: scope name
    :param decay:
    :param epsilon:
    :return: normalized input
    """
    with tf.variable_scope(scope):
        scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
        beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
        pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
        pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

        if is_training:
            if len(inputs.get_shape().as_list()) == 4:  # For 32D convolutional layers
                batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
            else:  # For fully-connected layers
                batch_mean, batch_var = tf.nn.moments(inputs, [0])
            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)
        else:
            return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)

def prelu(x, name=None):
    """
    Applies parametric leaky ReLU
    :param x: input tensor
    :param name: variable name
    :return: output tensor of the same shape
    """
    with tf.variable_scope(name_or_scope=name, default_name="prelu"):
        alpha = tf.get_variable('alpha', shape=x.get_shape()[-1], dtype=x.dtype,
                                initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, x) + alpha * tf.minimum(0.0, x)

def Relu(x):
    return tf.nn.relu(x)

def drop_out(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)


def concatenation(layers):
    return tf.concat(layers, axis=-1)