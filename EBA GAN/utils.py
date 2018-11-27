import tensorflow as tf
import numpy as np
import random

tf.set_random_seed(777)
np.random.seed(777)
random.seed(777)

initializer = tf.contrib.layers.variance_scaling_initializer(factor = 1.0)

def conv(inputs,filters,ker_size, strides, name):

    net = tf.layers.conv2d(inputs = inputs,
                           filters = filters,
                           kernel_size = [ker_size,ker_size],
                           strides = (strides,strides),
                           padding ="SAME",
                           kernel_initializer = initializer,
                           bias_initializer = tf.zeros_initializer(),
                           name = name,
                           reuse = tf.AUTO_REUSE)

    return net

def deconv(inputs, filters, ker_size, strides, name):

    net = tf.layers.conv2d_transpose(inputs = inputs,
                                     filters = filters,
                                     kernel_size = [ker_size, ker_size],
                                     strides = [strides, strides],
                                     kernel_initializer = initializer,
                                     bias_initializer = tf.zeros_initializer(),
                                     padding ="SAME",
                                     name = name,
                                     reuse = tf.AUTO_REUSE)

    return net

def upsample(inputs, size):

    net = tf.image.resize_nearest_neighbor(inputs, size=(size, size))

    return net

def maxpool(input,name):

    net = tf.nn.max_pool(value = input, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "SAME", name = name)

    return net

def bn(inputs,is_training, name):

    net = tf.contrib.layers.batch_norm(inputs, decay = 0.9, is_training = is_training, reuse = tf.AUTO_REUSE, scope = name)

    return net

def leaky(input):

    return tf.nn.leaky_relu(input)

def relu(input):

    return tf.nn.relu(input)

def drop_out(input, keep_prob):

    return tf.nn.dropout(input, keep_prob)

def dense(inputs, units, name):

    net = tf.layers.dense(inputs = inputs,
                          units = units,
                          reuse = tf.AUTO_REUSE,
                          name = name,
                          kernel_initializer = initializer,
                          bias_initializer=tf.zeros_initializer())

    return net

user_flags = []

def DEFINE_string(name, default_value, doc_string):

    tf.app.flags.DEFINE_string(name, default_value, doc_string)
    global user_flags
    user_flags.append(name)

def DEFINE_integer(name, default_value, doc_string):

    tf.app.flags.DEFINE_integer(name, default_value, doc_string)
    global user_flags
    user_flags.append(name)

def DEFINE_float(name, defualt_value, doc_string):

    tf.app.flags.DEFINE_float(name, defualt_value, doc_string)
    global user_flags
    user_flags.append(name)

def DEFINE_boolean(name, default_value, doc_string):

    tf.app.flags.DEFINE_boolean(name, default_value, doc_string)
    global user_flags
    user_flags.append(name)

def print_user_flags(line_limit = 100):

    print("-" * 80)

    global user_flags
    FLAGS = tf.app.flags.FLAGS

    for flag_name in sorted(user_flags):
        value = "{}".format(getattr(FLAGS, flag_name))
        log_string = flag_name
        log_string += "." * (line_limit - len(flag_name) - len(value))
        log_string += value
        print(log_string)

    return FLAGS


def salt_pepper_noise(images):

    size = np.shape(images)
    outputs = np.zeros(size, np.float32)
    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(size[2]):
                rdn = random.random()
                if rdn < 0.1:
                    outputs[i][j][k][:] = 0
                elif rdn > 0.9:
                    outputs[i][j][k][:] = 1
                else:
                    outputs[i][j][k][:] = images[i][j][k][:]

    return outputs

def change_diag_value(matrix, value):

    l = np.shape(matrix)[0]
    for i in range(l):
        matrix[i][i] = value

    return matrix


def make_patch(images, size, ksizes, strides):
    n_batch = size[0]
    W = size[1]
    H = size[2]
    C = size[3]
    W_n_patch = int((W - ksizes) / strides) + 1
    H_n_patch = int((H - ksizes) / strides) + 1

    imgs_croped = tf.reshape(images[0, 0: ksizes, 0: ksizes, :], [1, ksizes, ksizes, C])
    for i in range(n_batch):
        for j in range(W_n_patch):
            for k in range(H_n_patch):
                img_croped = tf.reshape(
                    images[i, strides * j: strides * j + ksizes, strides * k: strides * k + ksizes, :],
                    [1, ksizes, ksizes, C])
                imgs_croped = tf.concat([imgs_croped, img_croped], axis=0)
    imgs_croped = imgs_croped[1:]

    return imgs_croped