import tensorflow as tf


def label_generator_loss(lg_output,input_image):
    target_loss = tf.reduce_mean(tf.abs(input_image-lg_output))
    total_disc_loss = target_loss
    return total_disc_loss,target_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss+ generated_loss
    return total_disc_loss


def inference_generator_loss(disc_generated_output, ig_output, target,lambda_,ig_lv,lg_lv,alpha):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    gan_loss =loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.abs(target - ig_output)
    l1_loss = tf.reduce_mean(l1_loss)
    vector_loss = tf.reduce_mean(tf.abs(ig_lv - lg_lv))
    total_gen_loss = l1_loss*lambda_ + gan_loss + vector_loss*alpha
    return total_gen_loss, gan_loss, l1_loss,vector_loss

class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  #weight decay
  path = "number of data / batch_size"
  def __init__(self, initial_learning_rate):
    self.initial_learning_rate = initial_learning_rate
  def __call__(self, step):
    if (step < int(path*5)) :
      return self.initial_learning_rate
    elif step %int(path)==0:
      return self.initial_learning_rate*0.95
    else:
      return self.initial_learning_rate
