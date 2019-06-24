import tensorflow as tf
import tensorflow.contrib as tf_contrib

weight_init = tf.variance_scaling_initializer()
weight_regularizer = None
# weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
# weight_regularizer = None

##################################################################################
# Layer
##################################################################################

def conv(x, channels, kernel=4, stride=1, padding='same', use_bias=True, scope='conv'):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d(inputs=x, filters=channels, padding = padding, kernel_size=kernel, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, strides=stride, use_bias=use_bias)

    return x

def deconv(x, channels, kernel=3, stride=2, use_bias=True, scope='deconv_0') :
    with tf.variable_scope(scope):
        x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                       kernel_size=kernel, kernel_initializer=weight_init,
                                       kernel_regularizer=weight_regularizer,
                                       strides=stride, use_bias=use_bias, padding='SAME')

    return x

def linear(x, units, use_bias=True, scope='linear'):
    with tf.variable_scope(scope):
        x = flatten(x)
        x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)

    return x

def flatten(x) :
    return tf.layers.flatten(x)

def resize(image, scale):
	num_imgs,height,width = image.shape[0],image.shape[1],image.shape[2]
	new_width = int(width * scale)
	new_height = int(height * scale)
	curr_img = image
	curr_img = tf.image.resize_bicubic(curr_img, [new_height, new_width])
	return curr_img

##################################################################################
# Residual-block
##################################################################################

def resblock(x_init, channels, use_bias=True, scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = instance_norm(x)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = instance_norm(x)

    return x + x_init

##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.01):
    # pytorch alpha is 0.01
    return tf.nn.leaky_relu(x, alpha)

def relu(x):
    return tf.nn.relu(x)

def tanh(x):
    return tf.tanh(x)

##################################################################################
# Normalization function
##################################################################################

def instance_norm(x, scope='instance_norm'):
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)

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


def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))

    return loss
