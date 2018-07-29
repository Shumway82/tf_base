import tensorflow as tf
from tensorflow.python.framework import ops


def get_activation(name='selu'):
    """ Select the a activation function

        # Arguments
            name: Name of the activation function
                  (selu, relu, relu6, crelu, lrelu, elu, swish, swish_e, linear)

        # Returns
             Function pointer for a activation function.
    """

    dic_activation = {'selu': tf.nn.selu,
                      'relu': tf.nn.relu,
                      'relu6': tf.nn.relu6,
                      'crelu': tf.nn.crelu,
                      'lrelu': tf.nn.leaky_relu,
                      'elu': tf.nn.elu,
                      'swish': swish,
                      'swish_e': e_swish,
                      'linear': linear}
    try:
        activation = dic_activation[name]
        print(' [*] Activation:', activation.__name__)
        return activation
    except KeyError:
        raise KeyError(' [!] Activation: not found')


def swish(x):
    """ Swish activation function discribed in

        SEARCHING FOR ACTIVATION FUNCTIONS
        https://arxiv.org/pdf/1710.05941.pdf

        # Arguments
            x: Input tensor of shape `(rows, cols, channels)`

        # Input shape
            4D tensor with shape:
            `(batch, channels, rows, cols)` if data_format='channels_first'
            or 4D tensor with shape:
            `(batch, rows, cols, channels)` if data_format='channels_last'.

        # Output shape
            4D tensor with shape:
            `(batch, filters, new_rows, new_cols)` if data_format='channels_first'
            or 4D tensor with shape:
            `(batch, new_rows, new_cols, filters)` if data_format='channels_last'.
            `rows` and `cols` values might have changed due to stride.

        # Returns
             Output tensor of block.
        """
    return x * tf.nn.sigmoid(x)


def e_swish(x, beta=1.125):
    """ E-swish activation function discribed in

        E-swish: Adjusting Activations to Different Network Depths
        https://arxiv.org/pdf/1801.07145.pdf

    # Arguments
        x: Input tensor of shape `(rows, cols, channels)`
        beta: scaling value

    # Input shape
        4D tensor with shape:
        `(batch, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(batch, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.

    # Returns
         Output tensor of block.
    """
    return beta * x * tf.nn.sigmoid(x)


def lrelu(x, leak=0.2, name="lrelu"):
    """ Leaky-RELU activation function discribed in

        Empirical Evaluation of Rectified Activations in Convolution Network
        https://arxiv.org/pdf/1505.00853.pdf

        # Arguments
            x: Input tensor of shape `(rows, cols, channels)`
            leak: scaling value for negativ part of the function

        # Input shape
            4D tensor with shape:
            `(batch, channels, rows, cols)` if data_format='channels_first'
            or 4D tensor with shape:
            `(batch, rows, cols, channels)` if data_format='channels_last'.

        # Output shape
            4D tensor with shape:
            `(batch, filters, new_rows, new_cols)` if data_format='channels_first'
            or 4D tensor with shape:
            `(batch, new_rows, new_cols, filters)` if data_format='channels_last'.
            `rows` and `cols` values might have changed due to stride.

        # Returns
             Output tensor of block.
        """
    return tf.nn.leaky_relu(x, alpha=leak, name=name)


def selu(x, name="selu"):
    """ SELU activation function discribed in

        Self-Normalizing Neural Networks
        https://arxiv.org/pdf/1706.02515.pdf

        When using SELUs you have to keep the following in mind:
        (1) scale inputs to zero mean and unit variance
        (2) use SELUs
        (3) initialize weights with stddev sqrt(1/n)
        (4) use SELU dropout

        # Arguments
            x: Input tensor of shape `(rows, cols, channels)`
            leak: scaling value for negativ part of the function

         # Input shape
            4D tensor with shape:
            `(batch, channels, rows, cols)` if data_format='channels_first'
            or 4D tensor with shape:
             `(batch, rows, cols, channels)` if data_format='channels_last'.

        # Output shape
            4D tensor with shape:
            `(batch, filters, new_rows, new_cols)` if data_format='channels_first'
            or 4D tensor with shape:
            `(batch, new_rows, new_cols, filters)` if data_format='channels_last'.
            `rows` and `cols` values might have changed due to stride.

        # Returns
            Output tensor of block.
    """
    try:
        return tf.nn.selu(x, name)
    except AttributeError:
        with ops.name_scope(name):
            alpha = 1.6732632423543772848170429916717
            scale = 1.0507009873554804934193349852946
            return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))


slim = tf.contrib.slim


@slim.add_arg_scope
def prelu(x, scope, decoder=False):
    '''
    Performs the parametric relu operation. This implementation is based on:
    https://stackoverflow.com/questions/39975676/how-to-implement-prelu-activation-in-tensorflow

    For the decoder portion, prelu becomes just a normal prelu

    INPUTS:
    - x(Tensor): a 4D Tensor that undergoes prelu
    - scope(str): the string to name your prelu operation's alpha variable.
    - decoder(bool): if True, prelu becomes a normal relu.

    OUTPUTS:
    - pos + neg / x (Tensor): gives prelu output only during training; otherwise, just return x.

    '''
    # If decoder, then perform relu and just return the output
    if decoder:
        return tf.nn.relu(x, name=scope)

    alpha = tf.get_variable(scope + 'alpha', x.get_shape()[-1],
                            initializer=tf.constant_initializer(0.0),
                            dtype=tf.float32)
    pos = tf.nn.relu(x)
    neg = alpha * (x - abs(x)) * 0.5
    return pos + neg


def linear(x, *args):
    """ Linear activation function discribed in

        # Arguments
            x: Input tensor of shape `(rows, cols, channels)`

        # Input shape
            4D tensor with shape:
            `(batch, channels, rows, cols)` if data_format='channels_first'
            or 4D tensor with shape:
            `(batch, rows, cols, channels)` if data_format='channels_last'.

        # Output shape
            4D tensor with shape:
            `(batch, filters, new_rows, new_cols)` if data_format='channels_first'
            or 4D tensor with shape:
            `(batch, new_rows, new_cols, filters)` if data_format='channels_last'.
            `rows` and `cols` values might have changed due to stride.

        # Returns
            Output tensor of block.
        """
    return x
