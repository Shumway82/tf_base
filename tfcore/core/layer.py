import math
import numbers
import numpy as np
from keras.layers import Lambda
from tensorflow.contrib.layers import xavier_initializer_conv2d
from tensorflow.python.framework import ops, tensor_shape, tensor_util
from tensorflow.python.layers import utils
from tensorflow.python.ops import array_ops, math_ops, random_ops

from tfcore.core.activations import selu, lrelu, linear
from tfcore.core.normalization import *


def gaussian_noise_layer(input_layer, std):
    '''
    Add noise to a layer

    # Arguments:
        input_layer: a 3D or 4D Tensor of the input feature map or image.
        std: Standard deviation

    # Return :
        output(Tensor): a 4D Tensor that is in exactly the same size as the input input_layer
    '''
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    input_layer = tf.clip_by_value(input_layer + noise, -1.0, 1.0)
    return input_layer


def dropout(x, keep_prob, activation=linear, train=False):
    '''
    Dropout-layer for normal activation and selu-activation

    # Arguments:
        x: A 3D or 4D Tensor of the input feature map or image.x:
        keep_prob: A scalar Tensor with the same type as x. The probability that each element is kept.
        activation: Used activation-function. Is needed because selu have a different dropout-function
        train: Is train or not

    # Return :
        output(Tensor): a 4D Tensor that is in exactly the same size as the input input_layer
    '''
    if activation == selu:
        x = tf.cond(train,
                    lambda: dropout_selu(x, keep_prob, training=train),
                    lambda: x)
        print('Dropout SELU: ' + str(keep_prob))
    else:
        tf.layers.dropout(x, keep_prob, training=train)
        print('Dropout: ' + str(keep_prob))
    return x


def dropout_selu(x,
                 rate,
                 alpha=-1.7580993408473766,
                 fixed_point_mean=0.0,
                 fixed_point_var=1.0,
                 noise_shape=None,
                 seed=None,
                 name=None,
                 training=False):
    """
    Dropout to a value with rescaling.

    # Arguments:
        x: A 3D or 4D Tensor of the input feature map or image.x:
        rate: A scalar Tensor with the same type as x. The probability that each element is kept.
        alpha: Alpha-Value
        fixed_point_mean: Default 0
        fixed_point_var: Default 1.0
        noise_shape: Noise
        seed: Seed
        name: Name
        training: Is training

    # Return :
        output(Tensor): a 4D Tensor that is in exactly the same size as the input input_layer
    """

    def dropout_selu_impl(x, rate, alpha, noise_shape, seed, name):
        keep_prob = 1.0 - rate
        x = ops.convert_to_tensor(x, name="x")
        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                             "range (0, 1], got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        alpha = ops.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")
        alpha.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        if tensor_util.constant_value(keep_prob) == 1:
            return x

        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor + alpha * (1 - binary_tensor)

        a = math_ops.sqrt(fixed_point_var / (keep_prob * ((1 - keep_prob) *
                                                          math_ops.pow(alpha - fixed_point_mean,
                                                                       2) +
                                                          fixed_point_var)))

        b = fixed_point_mean - a * (keep_prob * fixed_point_mean + (1 - keep_prob) * alpha)
        ret = a * ret + b
        ret.set_shape(x.get_shape())
        return ret

    with ops.name_scope(name, "dropout", [x]) as name:
        return utils.smart_cond(training,
                                lambda: dropout_selu_impl(x, rate, alpha, noise_shape, seed, name),
                                lambda: array_ops.identity(x))


def spatial_dropout(x, p, seed, scope, is_training=True):
    """
    Performs a 2D spatial dropout that drops layers instead of individual elements in an input
     feature map.
    Note that p stands for the probability of dropping, but tf.nn.relu uses probability of keeping.

    ------------------
    Technical Details
    ------------------
    The noise shape must be of shape [batch_size, 1, 1, num_channels], with the height and width
    set to 1, because
    it will represent either a 1 or 0 for each layer, and these 1 or 0 integers will be
    broadcasted to the entire
    dimensions of each layer they interact with such that they can decide whether each layer
    should be entirely
    'dropped'/set to zero or have its activations entirely kept.
    --------------------------

    # Argumants
        x(Tensor): a 4D Tensor of the input feature map.
        p(float): a float representing the probability of dropping a layer
        seed(int): an integer for random seeding the random_uniform distribution that runs under
                   tf.nn.relu
        scope(str): the string name for naming the spatial_dropout
        is_training(bool): to turn on dropout only when training. Optional.

    # Return:
        output(Tensor): a 4D Tensor that is in exactly the same size as the input x,
                      with certain layers having their elements all set to 0 (i.e. dropped).
    """
    if is_training:
        keep_prob = 1.0 - p
        input_shape = x.get_shape().as_list()
        noise_shape = tf.constant(value=[input_shape[0], 1, 1, input_shape[3]])
        output = tf.nn.dropout(x, keep_prob, noise_shape, seed=seed, name=scope)

        return output

    return x


def variable_with_weight_decay(name,
                               shape,
                               activation=linear,
                               stddev=0.2,
                               initializer=None,
                               use_weight_decay=None,
                               is_conv_transposed=False,
                               trainable=True):
    """
    Init weights with decay

    # Arguments
        name: Weight name
        shape: Shape as 4D-Tensor
        activation: Activation to calculate the std
        stddev: default std
        use_weight_decay: Use weight decay
        is_conv_transposed: If conv_transpose is True input and output size will be changed
        trainable: Trainable True or False

    # Return
        output(Tensor): a 4D Tensor
    """

    # Determine number of input features from shape
    if is_conv_transposed:
        f_in = np.prod(shape[:-2]) if len(shape) == 4 else shape[0]
    else:
        f_in = np.prod(shape[:-1]) if len(shape) == 4 else shape[0]

    # Calculate sdev for initialization according to activation function
    if activation == tf.nn.selu:
        sdev = math.sqrt(1 / f_in)
    elif activation == tf.nn.relu or activation == tf.nn.leaky_relu or activation == tf.nn.relu6:
        sdev = math.sqrt(2 / f_in)
    elif activation == tf.nn.elu:
        sdev = math.sqrt(1.5505188080679277 / f_in)
    else:
        sdev = stddev

    if initializer is None:
        initializer = tf.truncated_normal_initializer(stddev=sdev, dtype=tf.float32)

    var = tf.get_variable(name=name,
                          shape=shape,
                         # regularizer=tf.contrib.layers.l1_l2_regularizer(scale_l1=1.0, scale_l2=1.0),
                          initializer=initializer,
                          trainable=trainable)

    if use_weight_decay is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), use_weight_decay, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def avg_pool(x, r=2, s=2):
    """
    Average polling-layer.

    # Arguments
        x: a 4D Tensor of the input feature map.
        r: A 1-D int Tensor of 4 elements. The size of the window for each dimension of the input tensor.
        s: A 1-D int Tensor of 4 elements. The stride of the sliding window for each dimension of the input tensor.

    # Return
        A Tensor of format specified by data_format. The max pooled output tensor.
    """
    return tf.nn.avg_pool(x, ksize=[1, r, r, 1], strides=[1, s, s, 1], padding="SAME")


def max_pool(x, r=2, s=2, name=None):
    """
        Max polling-layer.

        # Arguments
            x: a 4D Tensor of the input feature map.
            r: a 1-D int Tensor of 4 elements. The size of the window for each dimension of the input tensor.
            s: a 1-D int Tensor of 4 elements. The stride of the sliding window for each dimension of the input tensor.

        # Return
            A Tensor of format specified by data_format. The max pooled output tensor.
    """
    return tf.nn.max_pool(x, ksize=[1, r, r, 1], strides=[1, s, s, 1], padding="SAME", name=name)


def get_stddev(x, activation):
    """
    Calculate the Std for different feature-shape and activations.

    # Arguments
        x: a 4D Tensor of the input feature map.
        activation: Activation function

    # Return
        Std of corresponding activation and feature-shape
    """

    f_in = int(x.get_shape()[-1])

    if activation == tf.nn.selu:
        stddev = math.sqrt(1.0 / f_in)
    elif activation == tf.nn.relu or tf.nn.relu6:
        stddev = math.sqrt(2.0 / f_in)
    elif activation == lrelu:
        stddev = math.sqrt(2.0 / f_in)
    elif activation == tf.nn.elu:
        stddev = math.sqrt(1.5505188080679277 / f_in)
    else:
        stddev = math.sqrt(1.0 / f_in)
    return stddev


def get_var_maybe_avg(var_name, ema, **kwargs):
    ''' utility for retrieving polyak averaged params '''
    v = tf.get_variable(var_name, **kwargs)
    if ema is not None:
        v = ema.average(v)
    return v


def get_vars_maybe_avg(var_names, ema, **kwargs):
    ''' utility for retrieving polyak averaged params '''
    vars = []
    for vn in var_names:
        vars.append(get_var_maybe_avg(vn, ema, **kwargs))
    return vars


def conv2d(x,
           f_out,
           k_size=3,
           stride=1,
           activation=linear,
           use_bias=False,
           bias_init=0.0,
           init_scale=1.0,
           padding='SAME',
           use_weight_decay=False,
           normalization=None,
           use_spectral_norm=False,
           use_preactivation=False,
           get_weights=False,
           get_preactivation=False,
           set_weight=None,
           use_pixel_norm=False,
           use_wscaling=False,
           is_training=False,
           trainable=True,
           reuse=False,
           update_collection=SPECTRAL_NORM_UPDATE_OPS,
           name="conv2d"):
    shape = [k_size, k_size, int(x.get_shape()[-1]), f_out]
    w_count = shape[0] * shape[1] * shape[2] * shape[3]

    with tf.variable_scope(name):
        b = None
        if set_weight is None:
            w = variable_with_weight_decay('w',
                                           shape=shape,
                                           activation=activation,
                                           stddev=0.2,
                                           use_weight_decay=use_weight_decay,
                                           trainable=trainable)

            b = variable_with_weight_decay('biases',
                                           shape=[f_out],
                                           initializer=tf.constant_initializer(bias_init))
        else:
            w = set_weight

        if use_wscaling:
            scale = tf.reduce_mean(w ** 2)
            w = w / scale

        if use_preactivation:
            x = activation(x)

        if normalization is spectral_normed_weight or use_spectral_norm:
            if reuse:
                update_collection = 'NO_OPS'
            conv = tf.nn.conv2d(x,
                                spectral_normed_weight(w, update_collection=update_collection),
                                strides=[1, stride, stride, 1],
                                use_cudnn_on_gpu=True,
                                padding=padding)

        elif normalization is weight_norm:
            is_init = tf.Variable(True, name="init")
            g = tf.get_variable('g', shape=[f_out], dtype=tf.float32,
                                initializer=tf.constant_initializer(1.), trainable=True)

            def init(x, w, b, g):
                v_norm = tf.nn.l2_normalize(w, [0, 1, 2])
                x = tf.nn.conv2d(x, v_norm, strides=[1, stride, stride, 1], padding=padding)
                m_init, v_init = tf.nn.moments(x, [0, 1, 2])
                scale_init = init_scale / tf.sqrt(v_init + 1e-08)
                g = g.assign(scale_init)
                if use_bias:
                    b = b.assign(-m_init * scale_init)
                x = tf.reshape(scale_init, [1, 1, 1, f_out]) * (x - tf.reshape(m_init, [1, 1, 1, f_out]))
                is_init.assign(False)
                return x, g

            def train(x, w, b, g):
                # use weight normalization (Salimans & Kingma, 2016)
                W = tf.reshape(g, [1, 1, 1, f_out]) * tf.nn.l2_normalize(w, [0, 1, 2])

                # calculate convolutional layer output
                x = tf.nn.conv2d(x,
                                 W,
                                 strides=[1, stride, stride, 1],
                                 use_cudnn_on_gpu=True,
                                 padding=padding)

                return x, g

            conv, g = tf.cond(is_init,
                              lambda: init(x, w, b, g),
                              lambda: train(x, w, b, g))

        else:
            conv = tf.nn.conv2d(x,
                                w,
                                strides=[1, stride, stride, 1],
                                use_cudnn_on_gpu=True,
                                padding=padding)

        pre_activation = conv

        if use_wscaling:
            conv = conv * scale

        if use_bias:
            pre_activation = tf.nn.bias_add(conv, b)
            w_count += f_out

        if normalization is not None and \
                normalization is not spectral_normed_weight and \
                normalization is not weight_norm:
            pre_activation = normalization(pre_activation, training=is_training, reuse=reuse, scope=name)

        print(name + ': filter size: ' + str(k_size) + 'x' + str(k_size) + 'x' + str(
            conv.get_shape()[3]) + ' shape In=' + str(
            x.get_shape()) + ' shape Out=' + str(conv.get_shape()) + ' weights=' + str(w_count))

        if not use_preactivation:
            conv = activation(pre_activation)

        if use_pixel_norm:
            conv = pixel_norm(conv)

        if get_weights and use_bias:
            return conv, w, biases
        elif get_weights:
            return conv, w
        elif get_preactivation:
            return conv, pre_activation
        else:
            return conv


def deconv2d(x,
             f_out,
             output_shape=None,
             k_size=3,
             stride=1,
             activation=linear,
             use_bias=False,
             bias_init=0.0,
             padding='SAME',
             normalization=None,
             is_training=False,
             trainable=True,
             use_weight_decay=False,
             use_preactivation=False,
             get_weights=False,
             get_preactivation=False,
             set_weight=None,
             use_pixel_norm=False,
             use_wscaling=False,
             name="deconv2d"):
    shape = [k_size, k_size, f_out, int(x.get_shape()[-1])]
    w_count = shape[0] * shape[1] * shape[2] * shape[3]

    if output_shape is None:
        # if not is_training:
        # output_shape = [tf.shape(x)[0], tf.shape(x)[1] * stride,
        #                    tf.shape(x)[2] * stride, int(f_out)]
        # else:
        output_shape = [int(x.get_shape()[0]), int(x.get_shape()[1]) * stride,
                        int(x.get_shape()[2]) * stride, int(f_out)]

    with tf.variable_scope(name):

        if set_weight is None:
            w = variable_with_weight_decay('w',
                                           shape=shape,
                                           activation=activation,
                                           stddev=0.2,
                                           is_conv_transposed=True)
        else:
            w = set_weight

        if use_wscaling:
            w_scale = tf.reduce_mean(w ** 2)
            w = w / w_scale

        if use_preactivation:
            x = activation(x)

        deconv = tf.nn.conv2d_transpose(x,
                                        w,
                                        output_shape=output_shape,
                                        strides=[1, stride, stride, 1],
                                        padding=padding)
        pre_activation = deconv

        if use_wscaling:
            deconv = deconv * w_scale

        if use_bias:
            biases = variable_with_weight_decay('biases',
                                                shape=[f_out],
                                                initializer=tf.constant_initializer(bias_init))
            pre_activation = tf.nn.bias_add(deconv, biases)
            w_count += f_out

        if normalization is not None:
            pre_activation = normalization(pre_activation, training=is_training)

        print(name + ': filter size: ' + str(k_size) + 'x' + str(k_size) + 'x' + str(
            deconv.get_shape()[3]) + ' shape In=' + str(
            x.get_shape()) + ' shape Out=' + str(deconv.get_shape()) + ' weights=' + str(w_count))

        if not use_preactivation:
            deconv = activation(pre_activation)

        if use_pixel_norm:
            deconv = pixel_norm(deconv)

        if get_weights and use_bias:
            return deconv, w, biases
        elif get_weights:
            return deconv, w
        elif get_preactivation:
            return deconv, pre_activation
        else:
            return deconv


def dense_layer(x, f_out, bias_start=0.0, activation=linear, name='dense'):
    """
    Tensorflow Dense-Layer

    # Arguments
        x:  a 2D Tensor of the input features.
        f_out: Feature output-shape.
        bias_start: Init value for bias, Default 0.
        activation: Activation function.
        name: Name of the Layer

    # Return
        A 2D Tensor of the size of f_out.
    """

    stddev = get_stddev(x, activation)
    print(name + ': shape = ' + str(f_out) + ' stddev = ' + str("%.8f" % stddev))
    return tf.layers.dense(x, f_out,
                           activation=activation,
                           kernel_initializer=tf.random_normal_initializer(stddev=stddev),
                           bias_initializer=tf.constant_initializer(bias_start),
                           name=name)


def linear_layer(x,
                 f_out,
                 bias_start=0.0,
                 activation=linear,
                 with_w=False,
                 weights=None,
                 bias=None,
                 normalization=None,
                 use_weight_decay=False,
                 is_training=False,
                 reuse=False,
                 scope='linear'):
    """
    Linear Layer

    # Arguments
        x:  a 2D Tensor of the input features.
        f_out: Feature output-shape.
        bias_start: Init value for bias, Default 0.
        activation: Activation function.
        with_w: Get Weights as (x, weights, bias)
        weights: Set Weights. If not weigths will be created.
        bias: Set Bias. If non bias will be created.
        scope: Scope

    # Return
        A 2D Tensor of the size of f_out.
    """

    shape = [int(x.get_shape()[-1])]
    stddev = get_stddev(x, activation)

    with tf.variable_scope(scope):

        if weights is None:
            weights = variable_with_weight_decay('w',
                                                 shape=[shape[-1], f_out],
                                                 activation=activation,
                                                 stddev=stddev,
                                                 use_weight_decay=use_weight_decay,
                                                 trainable=True)
        if bias is None:
            bias = variable_with_weight_decay('biases',
                                              shape=[f_out],
                                              initializer=tf.constant_initializer(bias_start))
        pre_activation = tf.matmul(x, weights) + bias
        if normalization is not None and \
                normalization is not spectral_normed_weight and \
                normalization is not weight_norm:
            pre_activation = normalization(pre_activation, training=is_training, reuse=reuse, scope=scope)
        if with_w:
            return activation(pre_activation), weights, bias
        else:
            return activation(pre_activation)


def upscale_nearest_neighbor(x, f_size, is_training, resize_factor=2):
    """
    Implementation of nearest neightbor using conv_transposed().

    # Arguments
        x: a 4D Tensor of the input feature map.
        f_size: Filter-size
        is_training: Is training
        resize_factor: Scaling factor for width and height.

    # Return
        output(Tensor): a scaled 4D Tensor
    """
    identity = np.identity(f_size).astype(np.float32)
    identity = identity[None, None]
    x += 2
    x = deconv2d(x,
                 k_size=1,
                 f_out=f_size,
                 set_weight=identity,
                 stride=resize_factor,
                 is_training=is_training,
                 name='upscale_NN')
    x = max_pool(x, r=resize_factor, s=1)
    x -= 2
    return x


def rescale_layer(in_min, in_max, out_min, out_max, name=None):
    """ Returns a function/layer that rescale the input between out_min and out_max.

    # Arguments
        in_min: The minimum value of the input tensor.
        in_max: The maximum value of the input tensor.
        out_min: The minimum value of the output tensor.
        out_max: The maximum value of the output tensor.
        name: The name of the layer (affect the scope name).

    # Returns
        A keras layer that can then be called on a tensor.
    """
    in_min = float(in_min)
    in_max = float(in_max)
    out_min = float(out_min)
    out_max = float(out_max)
    a = (out_min - out_max) / (in_min - in_max)
    b = out_min - a * in_min
    return Lambda(lambda x: a * x + b, name=name)


def unpool(updates, mask, k_size=[1, 2, 2, 1], output_shape=None, scope=''):
    """
    Unpooling function based on the implementation by Panaetius at
     https://github.com/tensorflow/tensorflow/issues/2169

    # Arguments
        inputs(Tensor): a 4D tensor of shape [batch_size, height, width, num_channels]
                        that represents the input block to be upsampled
        mask(Tensor): a 4D tensor that represents the argmax values/pooling indices of the
                      previously max-pooled layer
        k_size(list): a list of values representing the dimensions of the unpooling filter.
        output_shape(list): a list of values to indicate what the final output shape should
                            be after unpooling
        scope(str): the string name to name your scope

    # Return
        ret(Tensor): the returned 4D tensor that has the shape of output_shape.
    """
    with tf.variable_scope(scope):
        mask = tf.cast(mask, tf.int32)
        input_shape = tf.shape(updates, out_type=tf.int32)
        #  calculation new shape
        if output_shape is None:
            output_shape = (input_shape[0], input_shape[1] * k_size[1], input_shape[2] * k_size[2],
                            input_shape[3])

        # calculation indices for batch, height, width and feature maps
        one_like_mask = tf.ones_like(mask, dtype=tf.int32)
        batch_shape = tf.concat([[input_shape[0]], [1], [1], [1]], 0)
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int32), shape=batch_shape)
        b = one_like_mask * batch_range
        y = mask // (output_shape[2] * output_shape[3])
        x = (mask // output_shape[3]) % output_shape[2]  # mask % (output_shape[2] *
        #  output_shape[3]) // output_shape[3]
        feature_range = tf.range(output_shape[3], dtype=tf.int32)
        f = one_like_mask * feature_range

        # transpose indices & reshape update values to one dimension
        updates_size = tf.size(updates)
        indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
        values = tf.reshape(updates, [updates_size])
        ret = tf.scatter_nd(indices, values, output_shape)
        return ret
