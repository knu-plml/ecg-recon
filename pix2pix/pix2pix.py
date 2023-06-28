import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Conv2D, Dense, Conv2DTranspose
from tensorflow.python.keras.models import Model
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
import time
import tensorflow.keras as tk
import tensorflow_addons as tfa
from biosppy.signals import tools as tools
import neural_structured_learning as nsl
from scipy import stats
from biosppy.signals import tools as tools
import neural_structured_learning as nsl
from scipy import stats
import sklearn.preprocessing as skp
import neural_structured_learning as nsl
from sklearn.utils import shuffle
from scipy import signal
from loss import *
time_len = 512
initializer = tf.random_normal_initializer(0., 0.02)


def load_generator():
    
    '''
    generator
    - based on the U-net architecture.
    - input shape: (16,512,1), output shape: (16,512,1).
    - input and output data contain two zero padding rows at the top and bottom, each.
    - 5 blocks in the encoder and the decoder, respectively.
    - 1 convolution layer for each block.
    - batch-normalization and leaky-ReLU after the convolution layer in the encoder, except the first block.
    - the decoder uses ReLU.

    initializer : for each layer, randomly select from (mean = 0, stdev = 0.02)
    encoder_inputs : input of the generator
    number_of_filter_encoder : the numbers of filters of the encoder in the generator
    number_of_filter_decoder : the numbers of filters of the decoder in the generator
    kernel : kernel
    stride : stride
    '''
    
    encoder_inputs = keras.Input(shape=(16, 512, 1), name='generator_input')
    number_of_filter_encoder = [64, 128, 256, 512, 1024]
    number_of_filter_decoder = [512, 256, 128, 64, 1]
    kernel = (2, 4)
    stride = [2, 2, 2, 2, (1, 2)]
    concatenate_encoder_block = []

    b_conv = tf.keras.layers.Conv2D(64, kernel, strides=stride[0], padding='same', kernel_initializer=initializer, use_bias=False)(encoder_inputs)  # 8
    b_output = tf.keras.layers.Activation('LeakyReLU')(b_conv)
    concatenate_encoder_block.append(b_output)

    for i in range(1, 5, 1):
        b_conv = tf.keras.layers.Conv2D(number_of_filter_encoder[i], kernel, strides=stride[i], padding='same', kernel_initializer=initializer, use_bias=False)(b_output)
        b_batch = tf.keras.layers.BatchNormalization()(b_conv)
        b_output = tf.keras.layers.Activation('LeakyReLU')(b_batch)
        if i == 4:
            break
        concatenate_encoder_block.append(b_output)

    b_conv = tf.keras.layers.Conv2DTranspose(512, (2, 4), strides=(1, 2), padding='same', kernel_initializer=initializer, use_bias=False)(b_output)
    b_batch = tf.keras.layers.BatchNormalization()(b_conv)
    b_output = tf.keras.layers.Activation('relu')(b_batch)

    for i in range(1, 4, 1):
        b_concat = tf.keras.layers.Concatenate()([b_output, concatenate_encoder_block[-i]])
        b_conv = tf.keras.layers.Conv2DTranspose(number_of_filter_decoder[i], kernel, strides=stride[4-i], padding='same', kernel_initializer=initializer, use_bias=False)(b_concat)
        b_batch = tf.keras.layers.BatchNormalization()(b_conv)
        b_output = tf.keras.layers.Activation('relu')(b_batch)

    b_concat = tf.keras.layers.Concatenate()([b_output, concatenate_encoder_block[-4]])
    encoder_outputs = layers.Conv2DTranspose(1, (2, 4), strides=(2), padding="same", kernel_initializer=initializer, use_bias=False)(b_concat)

    return keras.Model(encoder_inputs, encoder_outputs)


def load_discriminator():

    inp = tf.keras.layers.Input(shape=[16, time_len, 1], name='input_image')
    tar = tf.keras.layers.Input(shape=[16, time_len, 1], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])
    x = tf.keras.layers.Conv2D(64, (2, 4), strides=(2), kernel_initializer=initializer, padding='same', use_bias=False)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(128, (2, 4), strides=(2), kernel_initializer=initializer, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(256, (2, 4), strides=(2), kernel_initializer=initializer, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.ZeroPadding2D()(x)
    conv = tf.keras.layers.Conv2D(512, (2, 4), strides=(1), kernel_initializer=initializer, padding='same', use_bias=False)(x)
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
    last = tf.keras.layers.Conv2D(1, (2, 4), strides=(1), kernel_initializer=initializer, activation='sigmoid')(zero_pad2)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


def train_step(input_image, target, generator, discriminator, generator_optimizer, discriminator_optimizer, epoch, p):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target, p)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

    print('epoch {} gen_total_loss {} gen_gan_loss {} gen_l1_loss {}'.format(epoch, gen_total_loss, gen_gan_loss, gen_l1_loss))
