import tensorflow as tf
from tensorflow.contrib.layers import instance_norm, layer_norm, group_norm
import warnings


def pixel_norm(x):
    output = x / tf.sqrt(tf.reduce_mean(x ** 2, axis=3, keep_dims=True) + 1.0e-8)
    return output


def w_scaling_layer(conv, w):
    scale = tf.reduce_mean(w ** 2)
    conv = conv * scale
    return conv


def get_normalization(function_name):
    dic_normalization = {'BN': batch_norm_tf,
                         'IN': instance_norm_tf,
                         'SN': spectral_normed_weight,
                         'LN': layer_norm_tf,
                         'GN': group_norm_tf,
                         'WN': weight_norm,
                         'None': none_normalization}
    return dic_normalization.get(function_name, None)


SPECTRAL_NORM_UPDATE_OPS = "spectral_norm_update_ops"


def spectral_normed_weight(W, u=None, num_iters=1, update_collection=None, with_sigma=False):
    NO_OPS = 'NO_OPS'

    def _l2normalize(v, eps=1e-12):
        return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

    # Usually num_iters = 1 will be enough
    W_shape = W.shape.as_list()
    W_reshaped = tf.reshape(W, [-1, W_shape[-1]])
    if u is None:
        u = tf.get_variable("u", [1, W_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    def power_iteration(i, u_i, v_i):
        v_ip1 = _l2normalize(tf.matmul(u_i, tf.transpose(W_reshaped)))
        u_ip1 = _l2normalize(tf.matmul(v_ip1, W_reshaped))
        return i + 1, u_ip1, v_ip1

    _, u_final, v_final = tf.while_loop(
        cond=lambda i, _1, _2: i < num_iters,
        body=power_iteration,
        loop_vars=(tf.constant(0, dtype=tf.int32),
                   u, tf.zeros(dtype=tf.float32, shape=[1, W_reshaped.shape.as_list()[0]]))
    )
    if update_collection is None:
        warnings.warn(
            'Setting update_collection to None will make u being updated every W execution. This maybe undesirable'
            '. Please consider using a update collection instead.')
        sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
        # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
        W_bar = W_reshaped / sigma
        with tf.control_dependencies([u.assign(u_final)]):
            W_bar = tf.reshape(W_bar, W_shape)
    else:
        sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
        # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
        W_bar = W_reshaped / sigma
        W_bar = tf.reshape(W_bar, W_shape)
        # Put NO_OPS to not update any collection. This is useful for the second call of discriminator if the update_op
        # has already been collected on the first call.
        if update_collection != NO_OPS:
            tf.add_to_collection(update_collection, u.assign(u_final))
    if with_sigma:
        return W_bar, sigma
    else:
        return W_bar


def none_normalization(x, training=False, reuse=False, scope=None):
    return x


def weight_norm():
    print(' [*] Weight-Norm ')


def _instance_norm(x, training=False, reuse=False, scope=None):
    with tf.variable_scope('instance_norm', reuse=reuse):
        eps = 1e-5
        mean, sigma = tf.nn.moments(x, [1, 2], keep_dims=True)
        normalized = (x - mean) / (tf.sqrt(sigma) + eps)
        return normalized


def _instance_norm(x):
    """Adapted from https://github.com/hardikbansal/CycleGAN."""
    with tf.variable_scope("instance_norm"):
        epsilon = 1e-5
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable('scale', [x.get_shape()[-1]],
                                initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable('offset', [x.get_shape()[-1]],
                                 initializer=tf.constant_initializer(0.0))
        out = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset

        return out


def instance_norm_tf(x, training=False, reuse=False, scope=None):
    return instance_norm(x, epsilon=1e-5, reuse=False, scope=scope)


def batch_norm_tf(x, training=False, reuse=False, scope=None):
    return tf.layers.batch_normalization(x, momentum=0.9, reuse=reuse, training=training)


def group_norm_tf(x, training=False, reuse=False, scope=None):
    return group_norm(x, reuse=reuse)


def layer_norm_tf(x, training=False, reuse=False, scope=None):
    return layer_norm(x, reuse=False, scope=scope)


def normalize_weights(weights, values_range=1.0):
    return weights * values_range / tf.reduce_sum(weights)
