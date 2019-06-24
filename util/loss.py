import tensorflow as tf

def cyclic_loss_new(real, cycle):
    cost = tf.reduce_sum(tf.abs(real - cycle))/(2.0 * 64)
    return cost

def cyclic_loss(real, cycle):
    return tf.reduce_mean(tf.abs(real - cycle))

def lsgan_gen_loss(fake):
    return tf.reduce_mean(tf.squared_difference(fake, 1))

def lsgan_dis_loss(real, fake):
    return (tf.reduce_mean(tf.squared_difference(real, 1)) + 
            tf.reduce_mean(tf.squared_difference(fake, 0))) * 0.5

##################################################################################
# Loss function
##################################################################################

def discriminator_loss(type, real, fake):
    real_loss = 0
    fake_loss = 0

    if type == 'lsgan' :
        real_loss = tf.reduce_mean(tf.squared_difference(real, 1.0))
        fake_loss = tf.reduce_mean(tf.square(fake))

    if type == 'gan' :
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))

    if type == 'wgan':
        real_loss = tf.reduce_mean(fake) - tf.reduce_mean(real)
        fake_loss = 0

    loss = real_loss + fake_loss

    return loss


def generator_loss(type, fake):
    fake_loss = 0

    if type == 'lsgan' :
        fake_loss = tf.reduce_mean(tf.squared_difference(fake, 1.0))

    if type == 'gan' :
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))

    if type == 'wgan':
        fake_loss = -tf.reduce_mean(fake)

    loss = fake_loss


    return loss