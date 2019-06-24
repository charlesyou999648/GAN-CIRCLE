import tensorflow as tf
from ops import *

from models.layers import *

def resnet_block(x, nf, scope='res'):
    with tf.variable_scope(scope):
        y = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        y = conv2d(y, nf, 3, 1, padding='VALID', scope='_conv1')
        y = instance_norm(y, scope='_norm1')
        y = relu(y, name='_relu1')
        y = tf.pad(y, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        y = conv2d(y,nf, 3, 1, padding='VALID', scope='_conv2')
        y = instance_norm(y, scope='_norm2')
        return x + y

def generator(x, nf=32, c=1, scope='gen'):
    channel = 64
    layers = 12
    scale = 2
    min_filters = 16
    filters_decay_gamma = 1.5
    dropoutRate = 0.8
    padding = 'same'
    reuse=tf.AUTO_REUSE
    if scope == 'g_B':
        pixel_shuffler = True
        G_B = False
        n_stride = 2
    else:
        pixel_shuffler = True
        G_B = False
        n_stride = 2
        # channel = channel//2 
    he_init = tf.variance_scaling_initializer()
    
    with tf.variable_scope(scope, reuse=reuse):
        #bicubic image
        if G_B:
            bicubic_img = resize(x, scale)
        channel = channel * n_stride

        #feature map
        feature = conv(x, channel, kernel=3, stride=n_stride, scope='conv_0')   
        feature = lrelu(feature, 0.1)
        #print("feature shape", feature.shape)

        if dropoutRate < 1:
            feature = tf.layers.dropout(feature, dropoutRate)
        inputs = tf.identity(feature)
        #print("inputs shape", inputs.shape)

        #Dense layers
        for i in range(1, layers):
            if min_filters != 0 and i > 0:
                x1 = i / float(layers -1)
                y1 = pow(x1, 1.0 / filters_decay_gamma)
                output_feature_num = int((channel - min_filters) * (1 - y1) + min_filters)

            inputs = conv(inputs, output_feature_num, kernel=3,scope='conv'+str(i))
            # inputs = instance_norm(inputs, scope='ins_'+str(i))
            inputs = lrelu(inputs, 0.1)
            if dropoutRate < 1:
                inputs = tf.layers.dropout(inputs, dropoutRate)
            feature = tf.concat([feature, inputs], -1)
        
        #Reconstruction layers
        recons_a = tf.layers.conv2d(feature, 64, 1, padding='same', kernel_initializer=he_init,  name='A1', use_bias = True)
        recons_a = tf.nn.leaky_relu(recons_a, 0.1)

        if dropoutRate < 1:
            recons_a = tf.layers.dropout(recons_a, dropoutRate)
    
        recons_b1 = tf.layers.conv2d(feature, 32, 1, padding=padding, kernel_initializer=he_init, name='B1', use_bias = True)
        recons_b1 = tf.nn.leaky_relu(recons_b1, 0.1)
        if dropoutRate < 1:
            recons_b1 = tf.layers.dropout(recons_b1, dropoutRate)
    
        recons_b2 = tf.layers.conv2d(recons_b1, 8, 3, padding=padding, kernel_initializer=he_init, name='B2', use_bias = True)
        if dropoutRate < 1:
            recons_b = tf.layers.dropout(recons_b2, dropoutRate)
    
        recons = tf.concat([recons_a, recons_b], -1)
        num_feature_map = recons.get_shape()[-1]

        #building upsampling layer
        if pixel_shuffler:
            if scale == 4:
                recons_ps1 = pixelShuffler(recons, scale)
                recons = pixelShuffler(recons_ps1, scale)
            else:
                recons = tf.layers.conv2d(recons, num_feature_map, 3, padding=padding, kernel_initializer=he_init, name='Up-PS.conv0', use_bias = True)
                recons = tf.layers.conv2d_transpose(recons, num_feature_map//2, 4, strides = 2, padding=padding, kernel_initializer=he_init, name='Up-PS.T')
                recons = lrelu(recons, 0.1)
                
        out = tf.layers.conv2d(recons, 1, 3, padding=padding, kernel_initializer=he_init, name='R.conv0', use_bias = False)
        
        out = x + out

    return out

def discriminator(x, nf=64, scope='dis'):
    with tf.variable_scope(scope):
        #Convolutional layers
        d_c1 = conv2d(x, nf, 4, stride=2, scope='conv1')
        d_r1 = lrelu(d_c1, scope='lrelu1')
        
        d_c2 = conv2d(d_r1, nf * 2, 4, stride=2, scope='conv2')
        d_n2 = instance_norm(d_c2, scope='norm2')
        d_r2 = lrelu(d_n2, scope='lrelu2')
        
        d_c3 = conv2d(d_r2, nf * 4, 4, stride=2, scope='conv3')
        d_n3 = instance_norm(d_c3, scope='norm3')
        d_r3 = lrelu(d_n3, scope='lrelu3')
        
        d_c4 = conv2d(d_r3, nf * 8, 4, stride=1, scope='conv4')
        d_n4 = instance_norm(d_c4, scope='norm4')
        d_r4 = lrelu(d_n4, scope='lrelu4')
        
        d_c5 = conv2d(d_r4, 1, 4, stride=1, scope='conv5')
        
        return d_c5

