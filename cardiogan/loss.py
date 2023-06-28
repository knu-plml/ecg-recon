import tensorflow_addons as tfa
import tensorflow as tf
import keras
from tensorflow.keras import layers
import numpy as np
from biosppy.signals import tools as tools
from scipy import stats
import sklearn.preprocessing as skp
from scipy import signal
import librosa


def generator_loss(generated):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return loss_object(tf.ones_like(generated), generated)


def discriminator_loss(real, generated):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = loss_object(tf.ones_like(real), real)

    generated_loss = loss_object(tf.zeros_like(generated), generated)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 0.5


def calc_cycle_loss(real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

    return 5 * loss1
