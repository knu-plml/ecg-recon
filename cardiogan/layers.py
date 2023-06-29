import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
import numpy as np
from biosppy.signals import tools as tools
from scipy import stats
import sklearn.preprocessing as skp
from scipy import signal
import librosa


def downsample(ip, filter_size, kernel_size, norm, stride_size = 2):
    ip = Conv2D(filters = filter_size, kernel_size = kernel_size, strides = stride_size, padding = 'same', use_bias = False)(ip)
    if norm != 'none':
        ip = normalization(norm)(ip)
    ip = Activation(ip, activation = 'leaky_relu')

    return ip


def downsample_st1(ip, filter_size, kernel_size, norm, stride_size = 2):
    ip = Conv2D_st1(filters = filter_size, kernel_size = kernel_size, strides = stride_size, padding = 'same', use_bias = False)(ip)
    if norm != 'none':
        ip = normalization(norm)(ip)
    ip = Activation(ip, activation = 'leaky_relu')

    return ip


def upsample(ip, filter_size, kernel_size, norm, stride_size = 2, drop_rate = 0.5, apply_dropout = False):
    ip = DeConv2D(filters = filter_size, kernel_size = kernel_size, strides = stride_size, padding = 'same', use_bias = False)(ip)
    if norm != 'none':
        ip = normalization(norm)(ip)
    if apply_dropout:
        ip = Dropout(rate = drop_rate)
    ip = Activation(ip, activation = 'relu')

    return ip


def upsample_st1(ip, filter_size, kernel_size, norm, stride_size = 2, drop_rate = 0.5, apply_dropout = False):
    ip = DeConv2D_st1(filters = filter_size, kernel_size = kernel_size, strides = stride_size, padding = 'same', use_bias = False)(ip)
    if norm != 'none':
        ip = normalization(norm)(ip)
    if apply_dropout:
        ip = Dropout(rate = drop_rate)
    ip = Activation(ip, activation = 'relu')

    return ip


def Dense(units, activation = None):
    weights_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 0.02, seed = None)
    op = tf.keras.layers.Dense(units = units, activation = activation, use_bias = True, kernel_initializer = weights_initializer,
                               bias_initializer = 'zeros', kernel_regularizer = None, bias_regularizer = None, activity_regularizer = None,
                               kernel_constraint = None, bias_constraint = None)
    return op


def Conv2D(filters, kernel_size, strides = 2, padding = 'valid', activation = None, use_bias = True):
    weights_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 0.02, seed = None)
    op = tf.keras.layers.Conv2D(filters = filters, kernel_size = (2, kernel_size), strides = (strides), padding = padding, data_format = 'channels_last',
                                dilation_rate = 1, activation = None, use_bias = use_bias,
                                kernel_initializer = weights_initializer, bias_initializer = 'zeros',
                                kernel_regularizer = None, bias_regularizer = None, activity_regularizer = None,
                                kernel_constraint = None, bias_constraint = None)
    return op


def Conv2D_st1(filters, kernel_size, strides = 2, padding = 'valid', activation = None, use_bias = True):
    weights_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 0.02, seed = None)
    op = tf.keras.layers.Conv2D(filters = filters, kernel_size = (2, kernel_size), strides = (1, strides), padding = padding, data_format = 'channels_last',
                                dilation_rate = 1, activation = None, use_bias = use_bias,
                                kernel_initializer = weights_initializer, bias_initializer = 'zeros',
                                kernel_regularizer = None, bias_regularizer = None, activity_regularizer = None,
                                kernel_constraint = None, bias_constraint = None)
    return op


def DeConv2D(filters, kernel_size, strides = 2, padding = 'valid', use_bias = True):
    weights_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 0.02, seed = None)
    op = tf.keras.layers.Conv2DTranspose(filters = filters, kernel_size=(2, kernel_size), strides = (strides), padding = padding,
                                         output_padding = None, data_format = None, dilation_rate = (1, 1), activation = None, use_bias = use_bias,
                                         kernel_initializer = weights_initializer, bias_initializer = 'zeros', kernel_regularizer = None,
                                         bias_regularizer = None, activity_regularizer = None, kernel_constraint = None, bias_constraint = None)
    return op


def DeConv2D_st1(filters, kernel_size, strides = 2, padding = 'valid', use_bias = True):

    weights_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 0.02, seed = None)
    op = tf.keras.layers.Conv2DTranspose(filters = filters, kernel_size=(2, kernel_size), strides = (1, strides), padding = padding,
                                         output_padding = None, data_format = None, dilation_rate = (1, 1), activation = None, use_bias = use_bias,
                                         kernel_initializer = weights_initializer, bias_initializer = 'zeros', kernel_regularizer = None,
                                         bias_regularizer = None, activity_regularizer = None, kernel_constraint = None, bias_constraint = None)
    return op


def BatchNormalization(trainable = True, virtual_batch_size = None):
    op = tf.keras.layers.BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.001, center = True, scale = True,
                                            beta_initializer = 'zeros', gamma_initializer = 'ones',
                                            moving_mean_initializer = 'zeros', moving_variance_initializer = 'ones',
                                            beta_regularizer = None, gamma_regularizer = None, beta_constraint = None,
                                            gamma_constraint = None, renorm = False, renorm_clipping = None, renorm_momentum = 0.99,
                                            fused = None, trainable = trainable, virtual_batch_size = virtual_batch_size, adjustment = None, name = None)
    return op


def Activation(x, activation):
    if activation == 'relu':
        return tf.keras.activations.relu(x, alpha = 0.0, max_value = None, threshold = 0)
    elif activation == 'leaky_relu':
        return tf.keras.activations.relu(x, alpha = 0.2, max_value = None, threshold = 0)
    elif activation == 'sigmoid':
        return tf.keras.activations.sigmoid(x)
    elif activation == 'softmax':
        return tf.keras.activations.softmax(x, axis = -1)
    elif activation == 'tanh':
        return tf.keras.activations.tanh(x)
    else:
        raise ValueError('please check the name of the activation')


def Dropout(rate):
    op = tf.keras.layers.Dropout(rate = rate, noise_shape = None, seed = None)
    return op


def flatten():
    op = tf.keras.layers.Flatten(data_format = None)
    return op


def normalization(name):
    if name == 'none':
        return lambda: lambda x: x
    elif name == 'batch_norm':
        return keras.layers.BatchNormalization()
    elif name == 'instance_norm':
        return tfa.layers.InstanceNormalization()
    elif name == 'layer_norm':
        return keras.layers.LayerNormalization()


def attention_block_1d(curr_layer, conn_layer):
    Conv2D = tf.keras.layers.Conv2D
    """ adopted from https://github.com/lixiaolei1982/Keras-Implementation-of-U-Net-R2U-Net-Attention-U-Net-Attention-R2U-Net.-/blob/master/network.py
    """
    # theta_x(?,g_height,g_width,inter_channel)

    inter_channel = curr_layer.get_shape().as_list()[3]  # 4

    theta_x = Conv2D(inter_channel, 1, 1)(conn_layer)

    # phi_g(?,g_height,g_width,inter_channel)

    phi_g = Conv2D(inter_channel, 1, 1)(curr_layer)

    # f(?,g_height,g_width,inter_channel)

    f = Activation(keras.layers.add([theta_x, phi_g]), 'relu')

    # psi_f(?,g_height,g_width,1)

    psi_f = Conv2D(1, 1, 1)(f)

    rate = Activation(psi_f, 'sigmoid')

    # rate(?,x_height,x_width)

    # att_x(?,x_height,x_width,x_channel)

    att_x = keras.layers.multiply([conn_layer, rate])

    return att_x


def spectrogram(x, y):
    Sxx_x = []
    Syy_y = []
    x = np.array(x)
    y = np.array(y)
    for i in range(32):
        zeros = np.zeros((129, 129, 12))
        zeros_y = np.zeros((129, 129, 12))
        for j in range(1, 12, 1):
            stft = librosa.stft(x[i][j, :, 0], n_fft = 256, hop_length = 4)
            magnitude = np.abs(stft)
            log_spectrogram = librosa.amplitude_to_db(magnitude)
            zeros[:, :, j-1] = log_spectrogram
            stft = librosa.stft(y[i][j, :, 0], n_fft = 256, hop_length = 4)
            magnitude = np.abs(stft)
            log_spectrogram = librosa.amplitude_to_db(magnitude)
            zeros_y[:, :, j-1] = log_spectrogram
        zeros_y = np.resize(zeros_y, (128, 128, 12))
        zeros = np.resize(zeros, (128, 128, 12))
        Syy_y.append(zeros_y)
        Sxx_x.append(zeros)
    return np.array(Sxx_x), np.array(Syy_y)
