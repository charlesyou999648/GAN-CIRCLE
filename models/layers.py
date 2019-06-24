import tensorflow as tf

relu = tf.nn.relu

def lrelu(x, leak=0.2, scope='lrelu'):
    with tf.variable_scope(scope):
        return tf.maximum(x, leak * x)

def conv2d(x, n_outputs, kernel_size, stride=1, padding='SAME', 
           activation_fn=None, scope='conv2d'):
    with tf.variable_scope(scope):
        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(0.0)
        return tf.contrib.layers.conv2d(x, n_outputs, kernel_size, stride, 
                                        padding, activation_fn=activation_fn, 
                                        weights_initializer=w_init, 
                                        biases_initializer=b_init)

def deconv2d(x, n_outputs, kernel_size, stride=1, padding='SAME', 
           activation_fn=None, scope='deconv2d'):
    with tf.variable_scope(scope):
        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(0.0)
        return tf.contrib.layers.conv2d_transpose(x, n_outputs, 
                                                  [kernel_size, kernel_size], 
                                                  [stride, stride], padding, 
                                                  activation_fn=activation_fn, 
                                                  weights_initializer=w_init, 
                                                  biases_initializer=b_init)

def bn(x, eps=1e-5, scope='batch_norm'):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, 
                                        epsilon=eps, scale=True, scope=scope)

def instance_norm(x, eps=1e-5, scope='instance_norm'):
    with tf.variable_scope(scope):
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable('scale', [x.get_shape()[-1]], 
            initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable('offset', [x.get_shape()[-1]], 
            initializer=tf.constant_initializer(0.0))
        out = scale * tf.div(x - mean, tf.sqrt(var + eps)) + offset
        return out

