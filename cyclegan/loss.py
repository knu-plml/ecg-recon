import tensorflow_addons as tfa
import tensorflow as tf
import keras
from tensorflow.keras import layers
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(generated):
    return loss_object(tf.ones_like(generated), generated)


def discriminator_loss(real, generated):
    real_loss = loss_object(tf.ones_like(real), real)

    generated_loss = loss_object(tf.zeros_like(generated), generated)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 0.5


def calc_cycle_loss(real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

    return loss1


def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return 0.5 * loss


def train_step(input_image, target, epoch, generator, discriminator,
               generator_optimizer, discriminator_optimizer, s,
               generator2, generator2_optimizer, p, discriminator2, discriminator2_optimizer):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as gedisc_tape, tf.GradientTape() as disc2_tape:
        fake_y = generator(input_image, training=True)
        cycled_x = generator2(fake_y, training=True)

        fake_x = generator(target, training=True)
        cycled_y = generator2(fake_x, training=True)

        # same_x and same_y are used for identity loss.
        same_x = generator2(input_image, training=True)
        same_y = generator(target, training=True)

        disc_real_x = discriminator(input_image, training=True)
        disc_real_y = discriminator2(target, training=True)

        disc_fake_x = discriminator(fake_x, training=True)
        disc_fake_y = discriminator2(fake_y, training=True)

        # calculate the loss
        gen_loss = generator_loss(disc_fake_y)
        gen2_loss = generator_loss(disc_fake_x)

        total_cycle_loss = calc_cycle_loss(input_image, cycled_x, p) + calc_cycle_loss(target, cycled_y, p)

        # Total generator loss = adversarial loss + cycle loss
        total_gen_loss = gen_loss + total_cycle_loss*10 + identity_loss(target, same_y, p)*10
        total_gen2_loss = gen2_loss + total_cycle_loss*10 + identity_loss(input_image, same_x, p)*10

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

    print('epoch {} total_gen_loss {} gen_loss {} total_cycle_loss {}'.format(s, total_gen_loss, gen_loss, total_cycle_loss))
