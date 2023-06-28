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
