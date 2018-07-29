import glob
import io
import json
import os
import re
import shutil
import zipfile
from collections import deque
from inspect import signature
from keras import backend as K

import numpy as np
import scipy
import tensorflow as tf
from tensorflow.contrib.framework.python.framework import checkpoint_utils


def get_patches(input, patch_size=64, stride=0.0):
    size = [1, patch_size,patch_size, 1]
    patch_stride = [1, int(patch_size * (1 - stride)), int(patch_size * (1.0 - stride)), 1]
    patches = tf.extract_image_patches(input, size, patch_stride, [1, 1, 1, 1], 'VALID')
    return tf.reshape(patches, [-1, patch_size, patch_size, 3])

def conv_cond_concat(x, y):
    """ Concatenate conditioning vector on feature map axis.
    # Arguments
        x: 4D-Tensor
        y: 4D-Tensor

    # Return
        4D-Tensor
    """


    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(3, [x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])


def add_noise(img_batch):
    """ Add noise to a tensor

    # Arguments
        img_batch: A batch of a images or feature-maps.

    # Return
        A 4D-Tensor
    """
    for i in range(img_batch.shape[0]):
        noise = tf.random_normal(shape=tf.shape(img_batch.shape[1:]), mean=0.0, stddev=0.02,
                                 dtype=tf.float32)
        noise = np.clip(noise, -1., 1.)
        img_batch[i, :] += noise
    img_batch = tf.clip_by_value(img_batch, -1.0, 1.0)
    return img_batch


def downscale(x, factor):
    """ Downsale a Tensor

    # Arguments
        x: A 4D-Tensor
        factor: Scaling Factor as type of int

    # Return
        A downsceled 4D-Tensor
    """
    arr = np.zeros([factor, factor, 3, 3])
    arr[:, :, 0, 0] = 1.0 / factor ** 2
    weight = tf.constant(arr, dtype=tf.float32)
    downscaled = tf.nn.conv2d(x, weight, strides=[1, factor, factor, 1], padding='SAME')
    return downscaled


def normalize_weights(weights, values_range=1.0):
    """ Normalize weights to a definded value

    # Arguments
        weights: 2D-Tensor
        values_range: Normalize that the sum of weights corresponds to this value.

    # Return
        Normalized 2D-Tensor
    """
    return weights * values_range / tf.reduce_sum(weights)


def save_variables(sess, model_dir, scope='generator', global_step=0):
    model_dir = os.path.join(model_dir, 'TrainableVariables', scope)
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    values = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    vars = {}
    for value in values:
        name = value.name
        name = name.replace(':', '=')
        name = name.replace('/', '-')
        vars.update({name: value})
    saver = tf.train.Saver(vars)
    saver.save(sess, os.path.join(model_dir, scope), global_step=global_step)


def load_variable(sess, value, model_dir):
    model_dir = os.path.join(model_dir, 'TrainableVariables')
    value_files = sorted(glob.glob(os.path.join(model_dir, "*")))
    if len(value_files) == 0:
        return
    name = value.name
    name = name.replace(':', '=')
    name = name.replace('/', '-')

    model_dir = os.path.join(model_dir, name)
    if not os.path.exists(model_dir):
        print('Variable ' + value.name + ' not existing...')
        return

    saver = tf.train.Saver({name: value})
    saver.restore(sess, model_dir)
    print('Variable ' + value.name + ' loaded...')


def save_model(sess, model_dir, model_name, scope='generator', global_step=0):
    model_dir = os.path.join(model_dir, model_name)
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope))
    saver.save(sess, os.path.join(model_dir, scope), global_step=global_step, write_meta_graph=True,
               write_state=True)
    print(" [*] Model saving SUCCESS - " + os.path.join(model_dir, scope))
    return model_dir


def load_model(sess, model_dir, model_name, scope='generator'):
    model_dir = os.path.join(model_dir, model_name)

    ckpt = tf.train.get_checkpoint_state(model_dir)
    try:
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            vars_model = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

            vars_ckpt = checkpoint_utils.list_variables(os.path.join(model_dir, ckpt_name))

            vars_in_model = [var.name.split(':')[0] for var in vars_model]
            vars_in_ckpt = [var[0] for var in vars_ckpt]
            vars_to_remove = []
            for var in vars_in_model:
                if var not in vars_in_ckpt:
                    print(' [!] ' + var + ' not exists')
                    for i in range(len(vars_model)):
                        if vars_model[i].name.split(':')[0] == var:
                            vars_to_remove.append(vars_model[i])
            for var in vars_to_remove:
                vars_model.remove(var)

            saver = tf.train.Saver(vars_model)
            saver.restore(sess, os.path.join(model_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Model load SUCCESS - " + os.path.join(model_dir, ckpt_name))
            return True, counter
    except Exception as err:
        print(" [!] Model load FAILED - " + os.path.abspath(model_dir) + ', ' + str(err))

    print(" [!] Model load FAILED - no checkpoint in " + os.path.abspath(model_dir))
    return True, 0


def save(sess, saver, checkpoint_dir, epoch, step, resize_factor):
    model_name = "Model_R" + str(resize_factor) + '-' + str(epoch)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if saver is None:
        saver = tf.train.Saver()
    saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)
    print(' [*] Checkpoint saved: ' + checkpoint_dir)


def load(sess, checkpoint_dir):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        epoch = 0  # int(find_between(ckpt_name, '-', '-'))
        print(' [*] Checkpoint loaded: ' + checkpoint_dir)
        return True, counter, epoch
    else:
        print(' [!] Checkpoint FAILED: ' + checkpoint_dir)
        return False, 0


def load_pretrained_model(sess, variables, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    print(' [*] Checkpoint: ' + checkpoint_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        restorer = tf.train.Saver(variables)
        restorer.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        epoch = 0  # int(find_between(ckpt_name, '-', '-'))
        print(" [*] Load SUCCESS")
        return True, counter, epoch
    else:
        print(" [!] Load FAILED")
        return False, 0, 0


def get_global_steps(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        print(" [*] Global Steps 0")
        return 0

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print(" [*] Global Steps " + str(counter))
        return counter
    else:
        print(" [!] Global Steps 0")
        return 0


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.


    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def normalize_array(batch):
    return np.asarray([normalize(image) for image in batch], dtype=np.float32)


def normalize(image, normalization_type='tanh'):
    """
    Normalize Image

    # Arguments
        image: A `Tensor` of type `float32` or `float64`.
        activation: A value of type 'str'. If 'tanh' = [-1, 1],
        'sigmoid' = [0, 1], 'selu' mean = 0 and stddev = 1
        return: A `Tensor` of type `float32` or `float64`.
    # Return
        Normalized Tensor
    """
    if normalization_type is 'sigmoid':
        return np.asarray(image / 255.0, dtype=np.float32)
    elif normalization_type is 'selu':
        return np.array(
            [(image[i] - image[i].mean()) / image[i].std() for i in range(image.shape[0])])
    else:
        return np.asarray(image / 127.5 - 1.0, dtype=np.float32)


def inormalize(image, dtype=np.float32):
    return np.asarray((image + 1.0) * 127.5, dtype=dtype)


def save_images(images, size, image_path, normalized=False):
    num_im = size[0] * size[1]
    if normalized:
        return imsave(inverse_transform(images[:num_im]) * 255, size, image_path)
    else:
        return imsave(images[:num_im], size, image_path)


def inverse_transform(images):
    return (images + 1.) / 2.


def imsave(images, size=None, path=None):
    if size == [1, 1] or size is None:
        return scipy.misc.imsave(path, images[0])
    else:
        return scipy.misc.imsave(path, merge(images, size))


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    BW = size[0]
    BH = size[1]
    img = np.zeros((w * size[1], h * size[0], 3))
    idx = 0
    for bx in range(BW):
        for by in range(BH):
            img[int(by * h):int((by + 1) * h), int(bx * w):int((bx + 1) * w)] = images[idx]
            idx += 1

    return img


def save_experiment(dest_path, source_path='../../../tf_core/tfcore'):
    archiv_zip = zipfile.ZipFile(os.path.join(dest_path, 'experiment.zip'), 'w')

    for folder, subfolders, files in os.walk(source_path):

        for file in files:
            if file.endswith('.py'):
                archiv_zip.write(os.path.join(folder, file))

    archiv_zip.close()
    print(' [*] Experiment-ZIP saved at ' + source_path)


def save_config(dict, path, name='experiment'):
    path = os.path.join(path, name + '.json')
    with io.open(path, 'w', encoding='utf8') as outfile:
        str_ = json.dumps(dict,
                          indent=4, sort_keys=True,
                          separators=(',', ': '), ensure_ascii=False)
        outfile.writelines(str(str_))
    print(' [*] Config saved at ' + path)
    return


def pad_borders(net, k_size, mode="SYMMETRIC"):
    pad = int((k_size - 1) / 2)
    net = tf.pad(net, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode=mode)
    return net


class CLR:

    def __init__(self, sess, start_lr=0.0001, end_lr=0.00001, step_size=20, gamma=0.95, max_lex=50):
        self.sess = sess
        self.base_lr = start_lr
        self.max_lr = end_lr
        self.step_size = step_size
        self.gamma = gamma
        self.loss_averages_op = None
        self.loss_ema = None
        self.max_len = max_lex
        self.queue = deque(maxlen=self.max_len)
        self.moving_aves = deque(maxlen=self.max_len)
        self.max_derivativ = 0
        self.index_max_derivativ = 0
        self.learning_rate = start_lr

    def set_base_learning_rate(self, value):
        self.base_lr = value

    def set_max_learning_rate(self, value):
        self.max_lr = value

    def get_moving_average(self, value, iteration):
        self.queue.append(value)
        n = len(self.queue)
        if len(self.queue) > self.max_len:
            self.queue.popleft()
        if len(self.moving_aves) >= self.max_len - 1:
            self.moving_aves.popleft()
        cumsum = [0]
        for i, x in enumerate(list(self.queue), 1):
            cumsum.append(cumsum[i - 1] + x)
            if i >= n:
                moving_ave = (cumsum[i] - cumsum[i - n]) / n
                self.moving_aves.append(moving_ave)

        if len(self.moving_aves) > 30:
            derivativ = self.moving_aves[-20] - self.moving_aves[-1]
            if self.max_derivativ < derivativ:
                self.max_derivativ = derivativ
                self.index_max_derivativ = iteration
                self.base_lr = self.learning_rate
            print('loss_av: ' + str(derivativ) + ' max_derivativ ' + str(
                self.max_derivativ) + ' min_lr ' + str(self.base_lr) + ' min_lr_it ' + str(
                iteration))
        self.learning_rate += 0.00001
        return self.learning_rate

    def get_learning_rate(self, iteration, lr_type='exp_range'):
        """Given the inputs, calculates the lr that should be applicable for this iteration"""
        iteration += self.step_size
        cycle = np.floor(1 + iteration / (2 * self.step_size))
        x = np.abs(iteration / self.step_size - 2 * cycle + 1)
        lr = self.base_lr
        if lr_type == 'exp_range':
            scale = (self.gamma ** cycle)
            max_lr = self.max_lr * (1 / self.gamma)
            lr = self.base_lr + (max_lr - self.base_lr) * np.maximum(0, (1 - x)) * scale

        if lr_type == 'exp_range2':
            scale = (self.gamma ** cycle)
            a = (self.base_lr + (self.max_lr - self.base_lr) * scale)
            lr = a + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * scale

        return lr


def to_numpy(tf_function,
             input_shapes=(None, None, None, 3),
             input_types=None):
    """ This function convert a function using tensorflow tensors
    into a function using numpy arrays. The resulting function
    has the same signature as the input function.

    # Arguments
        tf_function: The tensorflow function to convert
        input_shapes: The information concerning the input shapes.
            You can be explicit by passing a list of shapes.
            Though is you pass only one shape, it's going to be
            assumed that the shape is the same for all inputs.
        input_types: The information concerning the input types.
            Defaults to `tf.float32` for compatibility.
            You can pass a list of types or just a single type.
            If you pass a single type, it will be assumed to be
            the same for all inputs.

    # Returns

    A function with the same signature as `tf_function` but can
    take numpy arrays as input.

    Example.
    ```python
    def dummy_tf(tensor1, tensor2):
        concatenation = tf.concat([tensor1, tensor2], 0)
        return concatenation, tensor2

    np_function = to_numpy(dummy_tf, (None, 2))

    arr1 = np.random.uniform((10,2))
    arr2 = np.random.uniform((10,2))

    result_concat, result_2 = np_function(arr1, arr2)
    ```
    """

    number_of_arguments = len((signature(tf_function)).parameters)

    if not isinstance(input_shapes, list):
        input_shapes = [input_shapes for _ in range(number_of_arguments)]

    if input_types is None:
        input_types = tf.float32
    if not isinstance(input_types, list):
        input_types = [input_types for _ in range(number_of_arguments)]

    placeholders = [tf.placeholder(dtype, shape)
                    for dtype, shape in zip(input_types, input_shapes)]

    outputs = tf_function(*placeholders)
    if isinstance(outputs, tuple):
        outputs = list(outputs)
    else:
        outputs = [outputs]

    numpy_function_lists = K.function(placeholders, outputs)

    def np_function(*args):
        output_list = numpy_function_lists(list(args))
        if len(output_list) == 1:
            return output_list[0]
        else:
            return tuple(output_list)
    return np_function


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


def reduce_std(x, axis=None, keepdims=False):
    """Standard deviation of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the standard deviation.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the standard deviation of elements of `x`.
    """
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))


def rgb_to_ycbcr(im, asuint8=False):
    """ A function to convert rgb images to YCbCr.

    # Arguments

        im: The rgb image to convert. Must be a Tensorflow
            tensor with positive values (0-255).
        asuint8:
            Set to `True` if you want a 8 bits tensor.
            If `False` returns a float32 tensor.

    # Returns

    An tensor with values between 0 and 255.
    """
    im = tf.cast(im, tf.float32)

    xform_rgb_to_ycbcr = tf.constant(np.array([[.299, .587, .114],
                                               [-.1687, -.3313, .5],
                                               [.5, -.4187, -.0813]], dtype=np.float32).T)

    ycbcr = K.dot(im, xform_rgb_to_ycbcr)
    tmp = ycbcr[..., 1:] + 128
    ycbcr = tf.concat([ycbcr[..., :1], tmp], axis=-1)
    if asuint8:
        ycbcr = tf.clip_by_value(ycbcr, 0, 255)
        ycbcr = tf.round(ycbcr)
        return tf.cast(ycbcr, tf.uint8)
    return ycbcr


def ycbcr_to_rgb(im, asuint8=False):
    """ A function to convert YCbCr images to rgb.

    # Arguments

        im: The YCbCr image to convert. Must be a Tensorflow
            tensor with positive values (0-255).
        asuint8:
            Set to `True` if you want a 8 bits tensor.
            If `False` returns a float32 tensor.

    # Returns

    A tensor with values between 0 and 255.
    """
    rgb = tf.cast(im, tf.float32)
    tmp = rgb[..., 1:] - 128
    rgb = tf.concat([rgb[..., :1], tmp], axis=-1)

    xform_ycbcr_to_rgb = tf.constant(np.array([[1, 0, 1.402],
                                               [1, -0.34414, -.71414],
                                               [1, 1.772, 0]], dtype=np.float32).T)

    rgb = K.dot(rgb, xform_ycbcr_to_rgb)
    if asuint8:
        rgb = tf.clip_by_value(rgb, 0, 255)
        rgb = tf.round(rgb)
        return tf.cast(rgb, tf.uint8)
    return rgb
