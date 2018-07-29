import numpy as np
import tensorflow as tf
from keras import backend as K

psnr = None#tf.image.psnr
ssim = None#tf.image.ssim
ssim_multiscale = None#tf.image.ssim_multiscale


def loss_normalization(loss, epsilon=1e-10):
    """
    Variable used for storing the scalar-value of the loss-function.
    # Arguments
        loss: Calculated unnormalized loss
        epsilon: A small value to prevent division by zero

    # Return
        Normalized loss
    """
    loss_value = tf.Variable(1.0)
    loss_value = loss_value.assign(loss)
    loss_normalized = loss / (loss_value + epsilon)

    return loss_normalized


def ls_discriminator(logit, label, factor=1.0):
    """ Objective for LS-GAN

    Paper: https://arxiv.org/pdf/1611.04076.pdf

    # Arguments
        logit: Logits of GAN
        label: True if real image else False
    # Return
        Loss as 1D-Tensor
    """
    if label is True:
        return tf.reduce_mean(tf.squared_difference(logit, 1)) * factor
    else:
        return tf.reduce_mean(tf.square(logit)) * factor

def ls_generator(logit):
    """ Objective for LS-GAN

    Paper: https://arxiv.org/pdf/1611.04076.pdf

    # Arguments
        logit: Logits of GAN
        label: True if real image else False
    # Return
        Loss as 1D-Tensor
    """
    return tf.reduce_mean(tf.squared_difference(logit, 1))


def get_true_labels(batch_size, values_range=0.2, soft=False, flipped=False):
    """
    Get True-Label for Discriminator.

    # Arguments
        batch_size: batch-size
        values_range: range for soft label
        soft: use soft label
        flipped: flip between real and fake label

    # Return
        A `Tensor` of the same type and shape as batch size.
    """
    if flipped:
        return get_fake_labels(batch_size, soft=soft)
    else:
        if soft:
            return tf.random_uniform([tf.shape(batch_size)[0], 1], 1.0 - values_range, 1.0)
        else:
            return tf.ones_like(batch_size)


def get_fake_labels(batch_size, values_range=0.2, soft=False):
    """
    Get Fake-Label for Discriminator.

    # Arguments
       batch_size: batch-size
        values_range: range for soft label
        soft: use soft label

    # Return
        A `Tensor` of the same type and shape as batch size.
        """
    if soft:
        return tf.random_uniform([tf.shape(batch_size)[0], 1], 0.0, values_range)
    else:
        return tf.zeros_like(batch_size)


def get_loss_function(name='MSE'):
    """
    Get Function-Pointer for loss-function.

    # Arguments
        name: Name of the loss-function. 'MSE', 'MAE', 'Charbonnier', 'L1', 'L2', 'Cosine', 'Dice', 'PSNR', 'DSSIM', 'HFEN'

    # Return
        Function-pointer
    """

    loss_dictionary = {'MSE': loss_mse,
                       'MAE': loss_mae,
                       'Charbonnier': loss_charbonnier,
                       'L1': loss_l1,
                       'L2': loss_l2,
                       'Cosine': loss_cosine,
                       'Dice': loss_dice,
                       'PSNR': psnr,
                       'DSSIM': dssim,
                       'HFEN': HFEN}

    return loss_dictionary.get(name, loss_none)


def loss_none(*args):
    return tf.constant(0, name='None_loss')


def tv_regularizer(x):
    return tf.reduce_sum(tf.image.total_variation(x, 'TV_Regularizer'))


def loss_dice(y, x):
    smooth = 1.
    y_true_f = K.flatten(y)
    y_pred_f = K.flatten(x)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def loss_charbonnier(y_real, x_out, alpha=0.45, epsilon=0.001):
    return tf.reduce_mean(tf.pow(tf.square(y_real - x_out) + tf.square(epsilon), alpha))


def loss_l2(x, y):
    return tf.reduce_sum(tf.square(y - x))


def loss_l1(x, y):
    return tf.reduce_sum(tf.abs(x - y))


def loss_mae(x, y):
    return tf.reduce_mean(tf.abs(x - y))


def loss_mse(y, x):
    return tf.losses.mean_squared_error(y, x)


def loss_cosine(y, x):
    return tf.losses.cosine_distance(tf.nn.l2_normalize(y, 0), tf.nn.l2_normalize(x, 0), dim=0)


def gram_matrix(x):
    """
    Gram-Matrix

    # Arguments
        x: a 4D-Tensor

    # Return
           a 4D-Tensor with same shape like input x
    """
    assert isinstance(x, tf.Tensor)
    b, h, w, ch = x.get_shape().as_list()
    features = tf.reshape(x, [b, h * w, ch])
    gram = tf.matmul(features, features, adjoint_a=True) / tf.constant(ch * w * h, tf.float32)
    return gram


def gram_matrix_v2(x):
    """
    Gram-Matrix

    # Arguments
        x: a 4D-Tensor

    # Return
        a 4D-Tensor with same shape like input x
    """
    shape = x.get_shape()

    # Get the number of feature channels for the input tensor,
    # which is assumed to be from a convolutional layer with 4-dim.
    num_channels = int(shape[3])

    # Reshape the tensor so it is a 2-dim matrix. This essentially
    # flattens the contents of each feature-channel.
    matrix = tf.reshape(x, shape=[-1, num_channels])

    # Calculate the Gram-matrix as the matrix-product of
    # the 2-dim matrix with itself. This calculates the
    # dot-products of all combinations of the feature-channels.
    gram = tf.matmul(tf.transpose(matrix), matrix)

    return gram


def HFEN(truth, predicted, loss_function=loss_mse):
    """
    High-Frequency-Error-Norm

    # Arguments
        truth: 4D-Tensor of ground truth.
        predicted: 4D-Tensor of generated image.
        loss_function: Loss-function

    # Return
        4D-Tensor with same shape as input.
    """

    a = np.array([[0, 0, 1, 0, 0],
                  [0, 1, 2, 1, 0],
                  [1, 2, -16, 2, 1],
                  [0, 1, 2, 1, 0],
                  [0, 0, 1, 0, 0]])

    a = tf.constant(a, dtype=tf.float32)
    a = a[:, :, None, None]
    a = tf.tile(a, [1, 1, 3, 1], name=None)

    x_pred = K.depthwise_conv2d(predicted, a)
    y_true = K.depthwise_conv2d(truth, a)

    loss = loss_function(x_pred, y_true)
    return loss


def reduce_var(x, axis=None, keepdims=False):
    """Variance of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the variance.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.


    # Returns
    A tensor with the variance of elements of `x`.
    """
    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)


def dssim(y_true, y_pred, max_value=2.0):
    return 1 / (K.epsilon() + ssim(y_true, y_pred, max_value) - 1)


def _to_db(value):
    return -10 * tf.log(1 - value) / np.log(10.)


def ssim_multiscale_db(img1, img2, max_val=2.0):
    """ Same as `ssim_multiscale`, but in decibels."""
    msssim_value = ssim_multiscale(img1, img2, max_val)
    return _to_db(msssim_value)


def ssim_db(img1, img2, max_val=2.0):
    """ Same as `ssim`, but in decibels."""
    ssim_value = ssim(img1, img2, max_val)
    return _to_db(ssim_value)
