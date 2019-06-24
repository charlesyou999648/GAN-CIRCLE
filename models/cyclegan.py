import numpy as np
import os
import random
import tensorflow as tf

from datetime import datetime
from glob import glob

import scipy.misc
import scipy.io

from models.models import *
from util.loss import *

class CycleGAN:
    def __init__(self, config,is_train, restore_ckpt, model_dir='./Network/cycleGAN/',summary_dir='./log/'):
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        
        #Image dimensions
        self.w, self.h, self.c = config.w,config.h,config.c
        self.ow, self.oh, self.oc = config.ow, config.oh, config.oc
        #Min/max voxels
        self.min, self.max = config.min_vox, config.max_vox
        
        #Training parameters
        self.restore_ckpt = restore_ckpt
        self.lambda_a = config.lambda_a
        self.lambda_b = config.lambda_b
        self.pool_size = config.pool_size
        self.base_lr_g = config.base_lr_g
        self.base_lr_d = config.base_lr_d
        self.max_step = config.epochs
        self.batch_size = config.batch


        self.model_dir = model_dir
        self.summary_dir = summary_dir

        #Initalize fake images
        self.fake_a = []
        self.fake_b = []
        
        #Randomly select training subjects to save

    def get_batch_data(self,data,idx):

        sub_data = data[idx:idx+self.batch_size]

        # Normalize values: -1 to 1

        # denom = (self.max - self.min) / 2
        # sub_data = (sub_data-self.min)/denom - 1

        # Randomly flip
        for i in range(self.batch_size):
            if np.random.choice(2,1)[0] == 1:
                sub_data[i] = sub_data[i,::-1]
            if np.random.choice(2,1)[0] == 1:
                sub_data[i] = sub_data[i,:,::-1]
        return sub_data

    def setup(self):
        self.real_a = tf.placeholder(tf.float32, 
                                     [None, None, None, self.c],
                                     name='input_a')
        self.real_b = tf.placeholder(tf.float32, 
                                     [None, None, None, self.oc],
                                     name='input_b')
        self.fake_pool_a = tf.placeholder(tf.float32, 
                                          [None, None, None, self.c],
                                          name='fake_pool_a')
        self.fake_pool_b = tf.placeholder(tf.float32, 
                                          [None, None, None, self.oc],
                                          name='fake_pool_b')
        
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.X1 = tf.placeholder(dtype=tf.float32, shape=[1, 512, 512, 1])
        self.n_fake_a = 0
        self.n_fake_b = 0
        self.lr_g = tf.placeholder(tf.float32, shape=[], name='lr_g')
        self.lr_d = tf.placeholder(tf.float32, shape=[], name='lr_d')
        self.forward()
    
    def forward(self):
        with tf.variable_scope('CycleGAN') as scope:
            #D(A), D(B)
            self.p_real_a = discriminator(self.real_a, scope='d_a')
            self.p_real_b = discriminator(self.real_b, scope='d_b')
            
            #G(A), G(B)
            self.fake_img_b = generator(self.real_a, c=self.oc, scope='g_a')
            self.fake_img_a = generator(self.real_b, c=self.c, scope='g_b')
            
            scope.reuse_variables()
            
            #D(G(B)), D(G(A))
            self.p_fake_a = discriminator(self.fake_img_a, scope='d_a')
            self.p_fake_b = discriminator(self.fake_img_b, scope='d_b')
            
            #G(G(A)), G(G(B))
            self.cycle_a = generator(self.fake_img_b, c=self.c, scope='g_b')
            self.cycle_b = generator(self.fake_img_a, c=self.oc, scope='g_a')

            scope.reuse_variables()
            
            self.p_fake_aa = generator(self.real_a, c=self.c, scope='g_b')
            self.p_fake_bb = generator(self.real_b, c=self.c, scope='g_a')

            scope.reuse_variables()
            
            #Fake pool for discriminator loss
            self.p_fake_pool_a = discriminator(self.fake_pool_a, scope='d_a')
            self.p_fake_pool_b = discriminator(self.fake_pool_b, scope='d_b')

            scope.reuse_variables()
            self.Y1 = generator(self.X1, c=1, scope='g_a')
 
            interpolates_a = alpha*tf.reshape(self.real_a, [self.batch_size, -1])+(1-alpha)*tf.reshape(self.fake_img_b, [self.batch_size, -1])
            interpolates_a = tf.reshape(interpolates_a, [self.batch_size, self.w, self.h, 1])
            gradients_a = tf.gradients(discriminator(interpolates_a, scope='d_a'), [interpolates_a])[0]
            slopes_a = tf.sqrt(tf.reduce_sum(tf.square(gradients_a), axis = 1))
            self.gradient_penalty_a = tf.reduce_mean((slopes_a-1.)**2)

            interpolates_b = alpha*tf.reshape(self.real_b, [self.batch_size, -1])+(1-alpha)*tf.reshape(self.fake_img_a, [self.batch_size, -1])
            interpolates_b = tf.reshape(interpolates_b, [self.batch_size, self.ow, self.oh, 1])
            gradients_b = tf.gradients(discriminator(interpolates_b, scope='d_b'), [interpolates_b])[0]
            slopes_b = tf.sqrt(tf.reduce_sum(tf.square(gradients_b), axis = 1))
            self.gradient_penalty_b = tf.reduce_mean((slopes_b-1.)**2)

    def loss(self):
        #Cycle consistency loss
        cyclic_loss_a = self.lambda_a * cyclic_loss_new(self.real_a, self.cycle_a)
        cyclic_loss_b = self.lambda_b * cyclic_loss_new(self.real_b, self.cycle_b)
        
        #LSGAN loss
        lsgan_loss_a = generator_loss(type = "wgan", fake = self.p_fake_a)
        lsgan_loss_b = generator_loss(type = "wgan", fake = self.p_fake_b)


        #Identity loss
        identity_loss_a = self.lambda_a * cyclic_loss_new(self.real_a, self.p_fake_aa)/2
        identity_loss_b = self.lambda_b * cyclic_loss_new(self.real_b, self.p_fake_bb)/2

        #Identity loss
        joint_loss_a = self.lambda_a / 10 * tf.reduce_sum(tf.image.total_variation(p_fake_aa))/20 + (1-self.lambda_a / 10) * tf.reduce_sum(tf.image.total_variation(real_a - p_fake_aa))/20
        joint_loss_b = self.lambda_b / 10 * tf.reduce_sum(tf.image.total_variation(p_fake_bb))/20 + (1-self.lambda_b / 10) * tf.reduce_sum(tf.image.total_variation(real_b - p_fake_bb))/20

        # #Supvision loss
        # sup_loss_b = 0.5*cyclic_loss_new(self.real_b,self.fake_img_b)
        # sup_loss_a = 0.5*cyclic_loss_new(self.real_a,self.fake_img_a)
        #Generator loss
        self.g_a_loss = cyclic_loss_a + cyclic_loss_b + lsgan_loss_b + identity_loss_a + joint_loss_a
        self.g_b_loss = cyclic_loss_b + cyclic_loss_a + lsgan_loss_a + identity_loss_b + joint_loss_b

        self.g_loss = self.g_a_loss + self.g_b_loss

        #Discriminator loss
        self.d_w_loss_a = discriminator_loss(type = "wgan", real = self.p_real_a, fake = self.p_fake_pool_a)
        self.d_w_loss_b = discriminator_loss(type = "wgan", real = self.p_real_b, fake = self.p_fake_pool_b)

        alpha = tf.random_uniform(
            shape=[self.batch_size, 1], 
            minval=0., 
            maxval=1
        )


        self.d_a_loss = self.d_w_loss_a + self.gradient_penalty_a
        self.d_b_loss = self.d_w_loss_b + self.gradient_penalty_b

        self.d_loss = self.d_a_loss + self.d_b_loss


        #Isolate variables
        self.vars = tf.trainable_variables()
        d_a_vars = [v for v in self.vars if 'd_a' in v.name]
        d_b_vars = [v for v in self.vars if 'd_b' in v.name]
        g_a_vars = [v for v in self.vars if 'g_a' in v.name]
        g_b_vars = [v for v in self.vars if 'g_b' in v.name]

        #Train while freezing other variables
        optimizer = tf.train.AdamOptimizer(self.lr_d, beta1=0.5)
        self.d_train = optimizer.minimize(self.d_loss, var_list=d_b_vars+d_a_vars)

        optimizer = tf.train.AdamOptimizer(self.lr_g, beta1=0.5)
        self.g_train = optimizer.minimize(self.g_loss, var_list=g_a_vars+g_b_vars)


        #Summary
        self.d_loss_summary = tf.summary.scalar('d_loss', self.d_loss)
        self.g_loss_summary = tf.summary.scalar('g_loss', self.g_loss)
    
    def fake_pool(self, fake, pool, n_fake):
        assert len(pool) <= self.pool_size
        results = []
        for i in range(self.batch_size):
            if n_fake < self.pool_size:
                # pool[n_fake] = fake[i]
                pool.append(fake[i])
                results.append(fake[i])
            else:
                p = random.random()
                if p < 0.5:
                    index = random.randint(0, self.pool_size - 1)
                    temp = pool[index]
                    pool[index] = fake[i]
                    results.append(temp)
                else:
                    results.append(fake[i])
            n_fake += 1
        return np.array(results)

    def train(self,data,label):

        self.setup()
        self.loss()

        total = len(data)

        init = (tf.global_variables_initializer(), 
                tf.local_variables_initializer())
        saver = tf.train.Saver(max_to_keep = 100)
        # ckpt_name = tf.train.latest_checkpoint(self.model_dir)
        # saver.restore(sess, ckpt_name)
        # saver.restore(sess,'./Network/cycleWGAN/cycleWGAN_100_1.ckpt')

        # test images
        input_data = scipy.io.loadmat('./test_mayo_20db_sz.mat')
        input_data = np.real(np.squeeze(input_data['LR']))
        #print("input_data shape",input_data.shape)
        input_data = np.expand_dims(input_data, axis=0)
        input_data = np.expand_dims(input_data, axis=3)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        with tf.Session(config=config) as sess:
            sess.run(init)
            
            if self.restore_ckpt:
                ckpt_name = tf.train.latest_checkpoint(self.model_dir)
                saver.restore(sess, ckpt_name)
            if not os.path.exists(self.summary_dir):
                os.makedirs(self.summary_dir)
            writer = tf.summary.FileWriter(self.summary_dir)

            print("Start training ... ")
            for epoch in range(sess.run(self.global_step), self.max_step):
                saver.save(sess, self.model_dir, global_step=epoch)
                
                if epoch < 100:
                    current_lr_g = self.base_lr_g
                    current_lr_d = self.base_lr_d
                else:
                    current_lr_g = self.base_lr_g - self.base_lr_g * (epoch - 100)/100
                    current_lr_d = self.base_lr_d - self.base_lr_d * (epoch - 100)/100

                for i in range(0,total,self.batch_size):
                    
                    batch_data = self.get_batch_data(data,i)
                    batch_label = self.get_batch_data(label,i)

                    #Optimize G
                    run_list = [self.g_train, self.fake_img_b, self.fake_img_a,
                                self.g_loss_summary,self.g_loss,self.g_b_loss,self.g_a_loss]
                    feed_dict = {self.real_a: batch_data, self.real_b: batch_label,
                                 self.lr_g: current_lr_g}
                    _, fake_b_temp,fake_a_temp, summary,g_loss,g_b_loss,g_a_loss = sess.run(run_list, feed_dict)
                    writer.add_summary(summary, epoch * total + i)


                    #Sample from fake  pool
                    fake_b_sample = self.fake_pool(fake_b_temp, self.fake_b,self.n_fake_b)
                    self.n_fake_b += self.batch_size
                    fake_a_sample = self.fake_pool(fake_a_temp, self.fake_a,self.n_fake_a)
                    self.n_fake_a += self.batch_size

                    #Optimize D_B
                    run_list = [self.d_train, self.d_loss_summary,self.d_loss,self.d_a_loss,self.d_b_loss,self.d_w_loss_a,self.d_w_loss_b]
                    feed_dict = {
                             self.real_a: batch_data, self.real_b: batch_label,
                             self.lr_d: current_lr_d,
                             self.fake_pool_b: fake_b_sample,
                             self.fake_pool_a: fake_a_sample
                            }
                    _, summary,d_loss,d_a_loss,d_b_loss,d_w_loss_a,d_w_loss_b = sess.run(run_list, feed_dict)
                    writer.add_summary(summary, epoch * total + i)

                    writer.flush()

                    if total-i <= self.batch_size:
                        print('Epoch: %d - gen_loss: %.6f - g_a_loss: %.6f - g_b_loss: %.6f - d_loss: %.6f - d_a_loss: %.6f - d_b_loss: %.6f - d_w_loss_a: %.6f - d_w_loss_b: %.6f ' % (epoch, g_loss, g_a_loss, g_b_loss, d_loss, d_a_loss, d_b_loss, d_w_loss_a, d_w_loss_b))
                        # print("Epoch: ",epoch)
                        # print("gen loss : ", g_loss)
                        # print("gen a loss : ", g_a_loss)
                        # print("gen b loss : ", g_b_loss)
                        # print("dis loss : ", d_loss)
                        # print("dis a loss : ", d_a_loss)
                        # print("dis b loss : ", d_b_loss)
                        
                np.random.shuffle(data)
                np.random.shuffle(label)
                    
                sess.run(tf.assign(self.global_step, epoch + 1))
                saver.save(sess, self.model_dir+'cycleWGAN_'+repr(epoch+1)+'_'+repr(i+1)+'.ckpt')
                # saver.save(sess, self.model_dir, global_step=epoch + 1)
                if not os.path.exists('./Test_Mayo_GAN_DCSCN_20db_new/'):
                    os.makedirs('./Test_Mayo_GAN_DCSCN_20db_new/')
                output_data = sess.run(self.Y1, feed_dict={self.X1: input_data})
                output_data = np.squeeze(output_data)
                #output_data = np.transpose(output_data, (1,0))
                scipy.misc.toimage(output_data, cmin=0.2333, cmax=0.9).save('./Test_Mayo_GAN_DCSCN_20db_new/dcscn_2D_'+repr(epoch+1)+'.png')
            writer.add_graph(sess.graph)
    
    def test(self,data,label):
        self.setup()

        saver = tf.train.Saver()
        init = (tf.global_variables_initializer(), 
                tf.local_variables_initializer())
        
        with tf.Session() as sess:
            sess.run(init)

            ckpt_name = tf.train.latest_checkpoint(self.model_dir)
            saver.restore(sess, ckpt_name)


            imgs = [self.fake_img_a,self.fake_img_b,self.cycle_a,self.cycle_b]


            b2a,a2b,aba,bab = sess.run(imgs,feed_dict={self.real_a: data, self.real_b: label})

            ## a :

        return b2a,a2b,aba,bab



















