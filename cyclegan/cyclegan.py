import tensorflow as tf
import keras
from tensorflow.keras import layers
from loss import *
initializer = tf.random_normal_initializer(0., 0.02)


def load_generator():
    '''
    CycleGAN generator
    - almost same as the Pix2Pix generator.
    - input shape: (16,512,1), output shape: (16,512,1).
    - input and output data contain two zero padding rows at the top and bottom, each
    - kernel size: (2,4)

    initializer : for each layer, randomly select from (mean = 0, stdev = 0.02)
    encoder_inputs : input of the generator
    number_of_filter_encoder : the number of filters of encoder in the encoder
    number_of_filter_decoder : the number of filters of decoder in the decoder
    kernel : kernel
    stride : stride
    '''
    encoder_inputs = keras.Input(shape=(16, 512, 1), name='generated_generator')

    number_of_filter_encoder = [64, 128, 256, 512, 1024]
    number_of_filter_decoder = [512, 256, 128, 64, 1]
    kernel = (2, 4)
    stride = [2, 2, 2, 2, (1, 2)]
    concatenate_encoder_block = []

    b_conv = tf.keras.layers.Conv2D(64, kernel, strides = stride[0], padding = 'same', kernel_initializer = initializer, use_bias = False)(encoder_inputs)
    b_output = tf.keras.layers.Activation('LeakyReLU')(b_conv)
    concatenate_encoder_block.append(b_output)

    for i in range(1, 5, 1):
        b_conv = tf.keras.layers.Conv2D(number_of_filter_encoder[i], kernel, strides = stride[i], padding = 'same', kernel_initializer = initializer, use_bias = False)(b_output)
        b_batch = tfa.layers.InstanceNormalization()(b_conv)
        b_output = tf.keras.layers.Activation('LeakyReLU')(b_batch)
        if i == 4:
            break
        concatenate_encoder_block.append(b_output)

    b_conv = tf.keras.layers.Conv2DTranspose(512, (2, 4), strides = (1, 2), padding = 'same', kernel_initializer = initializer, use_bias = False)(b_output)
    b_batch = tfa.layers.InstanceNormalization()(b_conv)
    b_output = tf.keras.layers.Activation('relu')(b_batch)

    for i in range(1, 4, 1):
        b_concat = tf.keras.layers.Concatenate()([b_output, concatenate_encoder_block[-i]])

        b_conv = tf.keras.layers.Conv2DTranspose(number_of_filter_decoder[i], kernel, strides = stride[4-i], padding = 'same', kernel_initializer = initializer, use_bias = False)(b_concat)
        b_batch = tfa.layers.InstanceNormalization()(b_conv)
        b_output = tf.keras.layers.Activation('relu')(b_batch)

    b_concat = tf.keras.layers.Concatenate()([b_output, concatenate_encoder_block[-4]])
    encoder_outputs = layers.Conv2DTranspose(1, (2, 4), strides = (2), padding = "same", kernel_initializer = initializer, use_bias = False)(b_concat)
    return tf.keras.Model(inputs = [encoder_inputs], outputs = encoder_outputs)


def load_discriminator():
    '''
    CycleGAN discriminator
    - based on the cyclegan discriminator.
    - uses 5 convolution layers (kernel size: (2,4))

    initializer : for each layer, randomly select from (mean = 0, stdev = 0.02)
    inp : input of the discriminator

    '''
    inp = tf.keras.layers.Input(shape=[16, 512, 1], name='input_image')

    x = tf.keras.layers.Conv2D(64, (2, 4), strides = (2), kernel_initializer = initializer, padding = 'same', use_bias = False)(inp)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(128, (2, 4), strides = (2), kernel_initializer = initializer, padding = 'same', use_bias = False)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(256, (2, 4), strides = (2), kernel_initializer = initializer, padding = 'same', use_bias = False)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.ZeroPadding2D()(x)
    conv = tf.keras.layers.Conv2D(512, (2, 4), strides = (1), kernel_initializer = initializer, padding = 'same', use_bias = False)(x)
    batchnorm1 = tfa.layers.InstanceNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
    last = tf.keras.layers.Conv2D(1, (2, 4), strides = (1), kernel_initializer = initializer, activation = 'sigmoid')(zero_pad2)

    return tf.keras.Model(inputs=[inp], outputs=last)


def train_step(input_image, target, generator, discriminator, generator_optimizer, discriminator_optimizer,
               epoch, generator2, generator2_optimizer, discriminator2, discriminator2_optimizer):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as gedisc_tape, tf.GradientTape() as disc2_tape:
        fake_y = generator(input_image, training = True)
        cycled_x = generator2(fake_y, training = True)

        fake_x = generator2(target, training = True)
        cycled_y = generator(fake_x, training = True)

        # same_x and same_y are used for identity loss.
        same_x = generator2(input_image, training = True)
        same_y = generator(target, training = True)

        disc_real_x = discriminator(input_image, training = True)
        disc_real_y = discriminator2(target, training = True)

        disc_fake_x = discriminator(fake_x, training = True)
        disc_fake_y = discriminator2(fake_y, training = True)

        # calculate the loss
        gen_loss = generator_loss(disc_fake_y)
        gen2_loss = generator_loss(disc_fake_x)

        total_cycle_loss = calc_cycle_loss(input_image, cycled_x) + calc_cycle_loss(target, cycled_y)

        # Total generator loss = adversarial loss + cycle loss
        total_gen_loss = gen_loss + total_cycle_loss * 10 + identity_loss(target, same_y) * 5
        total_gen2_loss = gen2_loss + total_cycle_loss * 10 + identity_loss(input_image, same_x) * 5

        disc_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc2_loss = discriminator_loss(disc_real_y, disc_fake_y)

    generator_gradients = gen_tape.gradient(total_gen_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)
    discriminator2_gradients = disc2_tape.gradient(disc2_loss,
                                                   discriminator2.trainable_variables)
    generator2_gradients = gedisc_tape.gradient(total_gen2_loss,
                                                generator2.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))
    generator2_optimizer.apply_gradients(zip(generator2_gradients,
                                             generator2.trainable_variables))
    discriminator2_optimizer.apply_gradients(zip(discriminator2_gradients,
                                                 discriminator2.trainable_variables))

    print('epoch {} total_gen_loss {} gen_loss {} total_cycle_loss {}'.format(epoch, total_gen_loss, gen_loss, total_cycle_loss))

