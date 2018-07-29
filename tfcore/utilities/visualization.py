import math
import numpy as np
import tensorflow as tf


def factorization(n):
    for i in range(int(np.math.sqrt(float(n))), 0, -1):
        if n % i == 0:
            if i == 1:
                print('Who would enter a prime number of filters')
            return i, int(n / i)


def get_grid_dim(x):
    """
    Transforms x into product of two integers
    :param x: int
    :return: two ints
    """
    factors = prime_powers(x)
    if len(factors) % 2 == 0:
        i = int(len(factors) / 2)
        return factors[i], factors[i - 1]

    i = len(factors) // 2
    return factors[i], factors[i]


def prime_powers(n):
    """
    Compute the factors of a positive integer
    Algorithm from https://rosettacode.org/wiki/Factors_of_an_integer#Python
    :param n: int
    :return: set
    """
    factors = set()
    for x in range(1, int(np.math.sqrt(n)) + 1):
        if n % x == 0:
            factors.add(int(x))
            factors.add(int(n // x))
    return sorted(factors)


def vis_conv(weight):
    cin = weight.shape.as_list()[2]
    cout = weight.shape.as_list()[3]
    W3_a = weight  # [8, 8, 32, 64]
    (grid_Y, grid_X) = factorization(cout)

    W3_b = W3_a
    W3_c = tf.split(W3_b, cout, 3)  # 64 x [8, 8, 32, 1]
    rows = []
    for i in range(grid_Y):
        row = tf.concat(W3_c[grid_X * i: grid_X * (i + 1)], 0)
        rows.append(row)
    W3_d = tf.concat(rows, 1)  # [64, 64, 32, 1]

    W3_e = tf.reshape(W3_d, [1, W3_d.shape[0], W3_d.shape[1], W3_d.shape[2]])
    W3_f = tf.split(W3_e, cin, 3)  # 32 x [1, 64, 64, 1]
    W3_g = tf.concat(W3_f[0:cin], 0)  # [32, 64, 64, 1]
    return W3_g


def put_kernels_on_grid(kernel, pad=1):
    """Visualize conv. filters as an image (mostly for the 1st layer).
      Arranges filters into a grid, with some paddings between adjacent filters.
      Args:
        kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
        pad:               number of black pixels around each filter (between them)
      Return:
        Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
    """

    # get shape of the grid. NumKernels == grid_Y * grid_X
    def factorization(n):
        for i in range(int(math.sqrt(float(n))), 0, -1):
            if n % i == 0:
                if i == 1:
                    print('Who would enter a prime number of filters')
                return i, int(n / i)

    (grid_Y, grid_X) = factorization(kernel.get_shape()[3].value)
    print('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)
    kernel = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x = tf.pad(kernel, tf.constant([[pad, pad], [pad, pad], [0, 0], [0, 0]]), mode='CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel.get_shape()[0] + 2 * pad
    X = kernel.get_shape()[1] + 2 * pad

    channels = kernel.get_shape()[2]

    # put NumKernels to the 1st dimension
    x = tf.transpose(x, (3, 0, 1, 2))
    # organize grid on Y axis
    x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))

    # switch X and Y axes
    x = tf.transpose(x, (0, 2, 1, 3))
    # organize grid on X axis
    x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

    # back to normal order (not combining with the next step for clarity)
    x = tf.transpose(x, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x = tf.transpose(x, (3, 0, 1, 2))
    x = tf.transpose(x, (3, 1, 2, 0))
    # scaling to [0, 255] is not necessary for tensorboard
    return x
