import tensorflow as tf
import keras
from loss import *
from layers import *
initializer = tf.random_normal_initializer(0., 0.02)


def load_generator():

    '''
    CardioGAN generator

    We modified CardioGAN from (https://github.com/pritamqu/ppg2ecg-cardiogan) to apply to our data .
    - input shape : (16,512,1), output shape : (16,512,1).
    - input and output data contain two zero padding rows at the top and bottom, each

    initializer : for each layer, randomly select from (mean = 0, stdev = 0.02)
    inputs : input of the generator
    filter size : the number of filters in the encoder and decoder
    '''

    filter_size = [64, 128, 256, 512, 512, 512]
    kernel_size = [16, 16, 16, 16, 16, 16]
    n_downsample = 6
    norm = 'layer_norm'
    skip_connection = True

    h = inputs = keras.Input(shape=(16, 512, 1))
    connections = []

    for k in range(n_downsample):
        if k == 0:
            h = downsample(h, filter_size[k], kernel_size[k], 'none')
        else:
            if k >= 3:
                h = downsample_st1(h, filter_size[k], kernel_size[k], norm)
            else:
                h = downsample(h, filter_size[k], kernel_size[k], norm)
        connections.append(h)

    # upsampling
    h = upsample(h, filter_size[k], kernel_size[k], norm, stride_size=1)
    if skip_connection:
        _h = attention_block_1d(curr_layer = h, conn_layer = connections[n_downsample-1])
        h = keras.layers.add([h, _h])

    for l in range(1, n_downsample):
        if l <= 3:
            h = upsample_st1(h, filter_size[k-l], kernel_size[k-l], norm)
            if skip_connection:
                _h = attention_block_1d(curr_layer = h, conn_layer = connections[k-l])
                h = keras.layers.add([h, _h])
        else:
            h = upsample(h, filter_size[k-l], kernel_size[k-l], norm)
            if skip_connection:
                _h = attention_block_1d(curr_layer = h, conn_layer = connections[k-l])
                h = keras.layers.add([h, _h])

    # output
    h = DeConv2D(filters = 1, kernel_size = kernel_size[k-l], strides = 2, padding = 'same')(h)
    h = Activation(h, activation='tanh')

    return keras.Model(inputs = inputs, outputs = h)


def load_time_discriminator():

    ''' time domain discriminator '''

    inp = tf.keras.layers.Input(shape = [16, 512, 1], name = 'input_image')
    x = tf.keras.layers.Conv2D(64, (2, 16), strides = 2, kernel_initializer = initializer, padding = 'same', use_bias = False)(inp)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(128, (2, 16), strides = 2, kernel_initializer = initializer, padding = 'same', use_bias = False)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(256, (2, 16), strides = 2, kernel_initializer = initializer, padding = 'same', use_bias = False)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.ZeroPadding2D()(x)
    conv = tf.keras.layers.Conv2D(512, (1, 16), strides = 1, kernel_initializer = initializer, padding = 'same', use_bias = False)(x)
    batchnorm1 = tf.keras.layers.LayerNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
    last = tf.keras.layers.Conv2D(1, (1, 16), strides = 1, kernel_initializer = initializer)(zero_pad2)
    return tf.keras.Model(inputs = [inp], outputs = last)


def load_frequency_discriminator():
    
    ''' frequency domain discriminator '''

    inp = tf.keras.layers.Input(shape=[128, 128, 12], name = 'input_image')
    x = tf.keras.layers.Conv2D(64, (7, 7), strides = 2, kernel_initializer = initializer, padding = 'same', use_bias = False)(inp)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(128, (7, 7), strides = 2, kernel_initializer = initializer, padding = 'same', use_bias = False)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(256, (7, 7), strides = 2, kernel_initializer = initializer, padding = 'same', use_bias = False)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.ZeroPadding2D()(x)
    conv = tf.keras.layers.Conv2D(512, (7, 7), strides = 2, kernel_initializer = initializer, padding = 'same', use_bias = False)(x)
    batchnorm1 = tf.keras.layers.LayerNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
    last = tf.keras.layers.Conv2D(1, (7, 7), strides = 2, kernel_initializer = initializer)(zero_pad2)
    return tf.keras.Model(inputs = [inp], outputs = last)


def train_step(real_x, real_y, generator_g, generator_f, discriminator_x, discriminator_y,
               generator_g_optimizer, generator_f_optimizer, discriminator_x_optimizer, discriminator_y_optimizer,
               epoch, discriminator_x2, discriminator_y2, discriminator_x2_optimizer, discriminator_y2_optimizer):
    with tf.GradientTape(persistent = True) as tape:
        fake_y = generator_g(real_x, training = True)
        cycled_x = generator_f(fake_y, training = True)

        fake_x = generator_f(real_y, training = True)
        cycled_y = generator_g(fake_x, training = True)

        spec_real_x, spec_real_y = spectrogram(real_x, real_y)
        spec_fake_x, spec_fake_y = spectrogram(fake_x, fake_y)

        disc_real_x = discriminator_x(real_x, training = True)
        disc_fake_y = discriminator_x2(fake_y, training = True)
        disc_spec_real_x = discriminator_y(spec_real_x, training = True)
        disc_spec_fake_y = discriminator_y2(spec_fake_y, training = True)

        disc_fake_x = discriminator_x(fake_x, training = True)
        disc_real_y = discriminator_x2(real_y, training = True)
        disc_spec_fake_x = discriminator_y(spec_fake_x, training = True)
        disc_spec_real_y = discriminator_y2(spec_real_y, training = True)

        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)

        gen_g_loss_fre = generator_loss(disc_spec_fake_y)
        gen_f_loss_fre = generator_loss(disc_spec_fake_x)

        total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

        total_gen_g_loss = 3 * gen_g_loss + total_cycle_loss + gen_g_loss_fre
        total_gen_f_loss = 3 * gen_f_loss + total_cycle_loss + gen_f_loss_fre

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_x2_loss = discriminator_loss(disc_real_y, disc_fake_y)
        disc_y_loss = discriminator_loss(disc_spec_real_x, disc_spec_fake_x)
        disc_y2_loss = discriminator_loss(disc_spec_real_y, disc_spec_fake_y)

    generator_g_gradients = tape.gradient(total_gen_g_loss,
                                          generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss,
                                          generator_f.trainable_variables)

    discriminator_x_gradients = tape.gradient(disc_x_loss,
                                              discriminator_x.trainable_variables)
    discriminator_x2_gradients = tape.gradient(disc_x2_loss,
                                               discriminator_x2.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss,
                                              discriminator_y.trainable_variables)
    discriminator_y2_gradients = tape.gradient(disc_y2_loss,
                                               discriminator_y2.trainable_variables)

    generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                              generator_g.trainable_variables))
    generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                              generator_f.trainable_variables))

    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                  discriminator_x.trainable_variables))
    discriminator_x2_optimizer.apply_gradients(zip(discriminator_x2_gradients,
                                                   discriminator_x2.trainable_variables))

    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                  discriminator_y.trainable_variables))
    discriminator_y2_optimizer.apply_gradients(zip(discriminator_y2_gradients,
                                                   discriminator_y2.trainable_variables))

    print('epoch {} total_gen_g_loss {} gen_g_loss {} total_gen_f_loss {} gen_f_loss {} total_cycle_loss {} disc_x_loss {}  disc_y_loss {} disc_x2_loss {} disc_y2_loss {}'.format(epoch, total_gen_g_loss, gen_g_loss, total_gen_f_loss, gen_f_loss, total_cycle_loss, disc_x_loss, disc_y_loss, disc_x2_loss, disc_y2_loss))
