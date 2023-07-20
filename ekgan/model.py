import tensorflow as tf
import keras
from tensorflow.keras import layers
from loss import *

initializer = tf.random_normal_initializer(0., 0.02)
time_len = 512

def load_inference_generator():
    '''
    EKGAN inference generator
    - input shape: (16,512,1), output shape: (16,512,1).
    - input and output data contain two zero padding rows at the top and bottom
    - kernel size: (2,4)

    initializer : for each layer, randomly select from (mean = 0, stdev = 0.02)
    ig_inputs : input of the generator
    filters_encoder : the number of filters of encoder
    filters_decoder : the number of filters of decoder
    kernel : kernel
    stride : stride
    '''
    filters_encoder = [64, 128, 256, 512, 1024]
    filters_decoder = [512, 256, 128, 64, 1]
    kernel = (2, 4)
    stride = [2, 2, 2, 2, (1, 2)]

    ig_inputs=keras.Input(shape = (16, time_len, 1), name = 'ig')
    h = []

    x = tf.keras.layers.Conv2D(filters_encoder[0], kernel, stride[0], padding = 'same', kernel_initializer = initializer, use_bias = False)(ig_inputs)
    x = tf.keras.layers.Activation('LeakyReLU')(x)
    h.append(x)
    for i in range(1,5,1):
        x = tf.keras.layers.Conv2D(filters_encoder[i], kernel, stride[i], padding = 'same', kernel_initializer = initializer, use_bias = False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('LeakyReLU')(x)
        h.append(x)
    ig_lv = x
    x = tf.keras.layers.Conv2DTranspose(filters_decoder[0], kernel, stride[-1], padding = 'same', kernel_initializer = initializer, use_bias = False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    for i in range(1,4,1):
        x = tf.keras.layers.Concatenate()([x, h[-i-1]])
        x = tf.keras.layers.Conv2DTranspose(filters_decoder[i], kernel, stride[-i-1], padding = 'same', kernel_initializer = initializer, use_bias = False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Concatenate()([x, h[0]])
    ig_outputs = layers.Conv2DTranspose(filters_decoder[4], kernel, stride[-4], padding = "same", kernel_initializer = initializer, use_bias = False)(x)
    inference_generator = keras.Model(inputs = [ig_inputs], outputs = [ig_outputs, ig_lv])
    inference_generator.summary()
    return inference_generator


def load_label_generator():
    '''
    EKGAN label generator
    - almost same as the inference generator, except for concatenation.
    - input shape: (16,512,1), output shape: (16,512,1).
    - input and output data contain two zero padding rows at the top and bottom
    - kernel size: (2,4)

    initializer : for each layer, randomly select from (mean = 0, stdev = 0.02)
    lg_inputs : input of the generator
    filters_encoder : the number of filters of encoder
    filters_decoder : the number of filters of decoder
    kernel : kernel
    stride : stride
    '''
    filters_encoder = [64, 128, 256, 512, 1024]
    filters_decoder = [512, 256, 128, 64, 1]
    kernel = (2, 4)
    stride = [2, 2, 2, 2, (1, 2)]

    lg_inputs=keras.Input(shape = (16, time_len, 1), name = 'lg')

    x = tf.keras.layers.Conv2D(filters_encoder[0], kernel, stride[0], padding = 'same', kernel_initializer = initializer, use_bias = False)(lg_inputs)
    x = tf.keras.layers.Activation('LeakyReLU')(x)

    for i in range(1,5,1):
        x = tf.keras.layers.Conv2D(filters_encoder[i], kernel, stride[i], padding = 'same', kernel_initializer = initializer, use_bias = False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('LeakyReLU')(x)
    lg_lv = x
    x = tf.keras.layers.Conv2DTranspose(filters_decoder[0], kernel, stride[-1], padding = 'same', kernel_initializer = initializer, use_bias = False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    for i in range(1,4,1):
        x = tf.keras.layers.Conv2DTranspose(filters_decoder[i], kernel, stride[-i-1], padding = 'same', kernel_initializer = initializer, use_bias = False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

    lg_outputs = layers.Conv2DTranspose(filters_decoder[4], kernel, stride[-4], padding="same",kernel_initializer = initializer, use_bias = False)(x)

    label_generator = keras.Model(inputs = [lg_inputs], outputs = [lg_outputs, lg_lv])
    label_generator.summary()
    return label_generator


def load_discriminator():
    '''
    EKGAN discriminator
    - uses 5 convolution layers (kernel size: (2,4))

    filters_encoder : the number of filters of encoder
    filters_decoder : the number of filters of decoder
    kernel : kernel
    stride : stride
    encoder_inputs : input of the discriminator (Input of inference generator)
    target : input of the discriminator (Original or generated image)
    '''
    filters_encoder = [32, 64, 128, 256, 512]
    filters_decoder = [256, 128, 64, 32, 1]
    kernel = [64, 32, 16, 8, 4]
    stride = [4, 4, 4, 2, 2]

    encoder_inputs = keras.Input(shape = (16, time_len, 1), name = 'dis')
    target = keras.Input(shape = (16, time_len, 1), name = 'dis_tar') 
    x = tf.keras.layers.Concatenate()([encoder_inputs, target])

    h = []
    x = tf.keras.layers.Conv2D(filters_encoder[0], (1, kernel[0]), (1, stride[0]), padding = 'same', kernel_initializer = initializer, use_bias = False)(x)
    x = tf.keras.layers.Activation('LeakyReLU')(x)
    h.append(x)
    for i in range(1,5,1):
        x = tf.keras.layers.Conv2D(filters_encoder[i], (1, kernel[i]), (1, stride[i]), padding = 'same', kernel_initializer = initializer, use_bias = False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('LeakyReLU')(x)
        h.append(x)

    x = tf.keras.layers.Conv2DTranspose(filters_decoder[0], (1, kernel[-1]), (1, stride[-1]), padding = 'same', kernel_initializer = initializer, use_bias = False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    for i in range(1,4,1):
        x = tf.keras.layers.Concatenate()([x, h[-i-1]])
        x = tf.keras.layers.Conv2DTranspose(filters_decoder[i], (1, kernel[-i-1]),(1, stride[-i-1]), padding = 'same', kernel_initializer = initializer, use_bias = False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Concatenate()([x, h[0]])
    encoder_outputs = layers.Conv2DTranspose(filters_decoder[4], (1, kernel[-5]), (1, stride[-4]), padding = "same", kernel_initializer = initializer, use_bias = False)(x)
    discriminator = keras.Model(inputs = [encoder_inputs, target],outputs = [encoder_outputs])
    discriminator.summary()
    return discriminator


def train_step(input_image, target, inference_generator, discriminator, inference_generator_optimizer, discriminator_optimizer, epoch, label_generator, label_generator_optimizer, lambda_, alpha):
    with tf.GradientTape() as ig_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as lg_tape:

        ig_output, ig_lv = inference_generator(input_image, training = True)
        lg_output, lg_lv = label_generator([input_image], training = True)

        disc_real_output = discriminator([input_image, target], training = True)
        disc_generated_output = discriminator([input_image, ig_output], training = True)

        total_lg_loss, lg_l1_loss = label_generator_loss(lg_output, input_image)

        total_ig_loss, ig_adversarial_loss, ig_l1_loss, vector_loss  = inference_generator_loss(disc_generated_output, ig_output, target, lambda_, ig_lv, lg_lv, alpha)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    inference_generator_gradients = ig_tape.gradient(total_ig_loss,
                                            inference_generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)
    label_generator_gradients = lg_tape.gradient(total_lg_loss,
                                                 label_generator.trainable_variables)

    inference_generator_optimizer.apply_gradients(zip(inference_generator_gradients,
                                            inference_generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))
    label_generator_optimizer.apply_gradients(zip(label_generator_gradients,
                                                label_generator.trainable_variables))

    print('epoch {} gen_total_loss {} ig_adversarial_loss {} ig_l1_loss {} lg_l2_loss {} vector_loss {}  '.format(epoch, total_ig_loss, ig_adversarial_loss, ig_l1_loss, lg_l1_loss, vector_loss))


