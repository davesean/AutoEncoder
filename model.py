from __future__ import division
import os
import time
from sys import stdout
from glob import glob
import tensorflow as tf
import numpy as np
import cv2
from six.moves import xrange

from ops import *

def add_noise(image, noise=0.1):
    with tf.name_scope("add_noise"):
        return image+tf.random_normal(shape=tf.shape(image), mean=0.0, stddev=noise, dtype=tf.float32)

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image/255 * 2 - 1
def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return ((image + 1) / 2)*255

class AutoEncoder(object):
    def __init__(self, sess, image_size=256,
                 batch_size=1, sample_size=1, output_size=256,
                 aef_dim=64, input_c_dim=3, output_c_dim=3, dataset_name='cityscapes_GAN',
                 checkpoint_dir=None, data=None, momentum=0.9, noise_std_dev=0.0):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [256]
            aef_dim: (optional) Dimension of AE filters in first conv layer. [64]
            input_c_dim: (optional) Dimension of input image color. For grayscale input, set to 1. [3]
            output_c_dim: (optional) Dimension of output image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_grayscale = (input_c_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size

        self.aef_dim = aef_dim

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim

        self.noise_std_dev=noise_std_dev
        # batch normalization : deals with poor initialization helps gradient flow

        self.ae_bn_e2 = batch_norm(name='ae_bn_e2', momentum=momentum)
        self.ae_bn_e3 = batch_norm(name='ae_bn_e3', momentum=momentum)
        self.ae_bn_e4 = batch_norm(name='ae_bn_e4', momentum=momentum)
        self.ae_bn_e5 = batch_norm(name='ae_bn_e5', momentum=momentum)
        self.ae_bn_e6 = batch_norm(name='ae_bn_e6', momentum=momentum)
        self.ae_bn_e7 = batch_norm(name='ae_bn_e7', momentum=momentum)
        self.ae_bn_e8 = batch_norm(name='ae_bn_e8', momentum=momentum)

        self.ae_bn_d1 = batch_norm(name='ae_bn_d1', momentum=momentum)
        self.ae_bn_d2 = batch_norm(name='ae_bn_d2', momentum=momentum)
        self.ae_bn_d3 = batch_norm(name='ae_bn_d3', momentum=momentum)
        self.ae_bn_d4 = batch_norm(name='ae_bn_d4', momentum=momentum)
        self.ae_bn_d5 = batch_norm(name='ae_bn_d5', momentum=momentum)
        self.ae_bn_d6 = batch_norm(name='ae_bn_d6', momentum=momentum)
        self.ae_bn_d7 = batch_norm(name='ae_bn_d7', momentum=momentum)

        self.data = data
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        # TODO check that only RGB is needed
        # Get the data descriptors with the shape of data coming
        data_description = data.get_data_description()
        data_description = [data_description[0], {
            key: [None, *description]
            for key, description in data_description[1].items()}]

        # Create an iterator for the data
        self.iter_handle = tf.placeholder(tf.string, shape=[],
                                              name='training_placeholder')
        iterator = tf.data.Iterator.from_string_handle(
            self.iter_handle, *data_description)
        training_batch = iterator.get_next()

        self.build_model(training_batch['rgb'])

    def build_model(self, input):
        # TODO check if pre + post process improves quality
        # self.real_A = preprocess(input)
        self.input = input

        self.output = self.autoencoder(self.input)

        # self.ae_loss = tf.norm((self.input-self.output))
        self.ae_loss = tf.reduce_mean(tf.abs(self.input - self.output))

        self.input_sum = tf.summary.image("Input", deprocess(self.input)[...,::-1])
        self.output_sum = tf.summary.image("Output", deprocess(self.output)[...,::-1])

        self.ae_loss_sum = tf.summary.scalar("ae_loss", self.ae_loss)

        t_vars = tf.trainable_variables()

        self.ae_vars = [var for var in t_vars if 'ae_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, args):
        """Train auto encoder"""

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            g_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                              .minimize(self.ae_loss, var_list=self.ae_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.sum = tf.summary.merge([self.ae_loss_sum, self.input_sum, self.output_sum])

        self.writer = tf.summary.FileWriter(self.checkpoint_dir, self.sess.graph)

        if args.checkpoint is not None and self.load(os.path.join(args.EXP_OUT,str(args.checkpoint))):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in range(0,1):
            input_data = self.data.get_trainset()
            data_iterator = input_data.repeat(args.max_epochs).batch(args.batch_size).make_one_shot_iterator()
            data_handle = self.sess.run(data_iterator.string_handle())
            counterTrain = 1
            start_time = time.time()
            while True:
                if np.mod(counterTrain, args.num_print) == 1:
                    try:
                        _, summary_str, ae_l = self.sess.run([g_optim, self.sum, self.ae_loss],
                                                       feed_dict={ self.iter_handle: data_handle })
                    except tf.errors.OutOfRangeError:
                        print("INFO: Done with all steps")
                        self.save(self.checkpoint_dir, counterTrain)
                        break

                    self.writer.add_summary(summary_str, counterTrain)
                    print("Step: [%2d] rate: %4.4f steps/sec, ae_loss: %.8f" \
                        % (counterTrain,args.batch_size*counterTrain/(time.time() - start_time), ae_l))
                    stdout.flush()
                else:
                    try:
                        self.sess.run(g_optim,feed_dict={ self.iter_handle: data_handle })
                    except tf.errors.OutOfRangeError:
                        print("INFO: Done with all training steps")
                        self.save(self.checkpoint_dir, counterTrain)
                        break
                counterTrain += 1

        pred_array = np.zeros((15,1))

        if not os.path.exists(os.path.join(args.file_output_dir,str(args.RUN_id))):
            os.makedirs(os.path.join(args.file_output_dir,str(args.RUN_id)))

        validation_data = self.data.get_validation_set()
        valid_iterator = validation_data.batch(args.batch_size).make_one_shot_iterator()
        valid_handle = self.sess.run(valid_iterator.string_handle())
        counter = 1
        while True:
            # Valide AE network
            try:
                ae_l, outImage = self.sess.run([self.ae_loss, self.output],
                                               feed_dict={ self.iter_handle: valid_handle })
            except tf.errors.OutOfRangeError:
                print("INFO: Done with all validation samples")
                break

            pred_array[counter-1] = ae_l
            filename = str(args.RUN_id)+"_reconstruction" + str(counter) + ".png"
            # cv2.imwrite(os.path.join(args.file_output_dir,str(args.RUN_id),filename), deprocess(outImage[0,:,:,:]), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            cv2.imwrite(os.path.join(args.file_output_dir,str(args.RUN_id),filename), outImage[0,:,:,:], [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            counter += 1
        print(pred_array)
        return pred_array


    def validate(self, args):
        """Validate auto encoder"""
        pred_array = np.zeros((15,2))
        counter = 1

        self.load(os.path.join(args.EXP_OUT,str(args.checkpoint)))

        if not os.path.exists(os.path.join(args.file_output_dir,str(args.checkpoint))):
            os.makedirs(os.path.join(args.file_output_dir,str(args.checkpoint)))

        for epoch in range(0,1):
            validation_data = self.data.get_validation_set()
            valid_iterator = validation_data.batch(args.batch_size).make_one_shot_iterator()
            valid_handle = self.sess.run(valid_iterator.string_handle())
            while True:
                # Valide AE network
                try:
                    ae_l, outImage = self.sess.run([self.ae_loss, self.output],
                                                   feed_dict={ self.iter_handle: valid_handle })
                except tf.errors.OutOfRangeError:
                    print("INFO: Done with all validation samples")
                    break

                pred_array[counter-1] = ae_l
                filename = str(args.checkpoint)+"_reconstruction" + str(counter) + ".png"
                # cv2.imwrite(os.path.join(args.file_output_dir,str(args.checkpoint),filename), deprocess(outImage[0,:,:,:]), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                cv2.imwrite(os.path.join(args.file_output_dir,str(args.checkpoint),filename), outImage[0,:,:,:], [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                counter += 1
            print(pred_array)
            return pred_array

    def autoencoder(self, image, y=None):
        with tf.variable_scope("autoencoder") as scope:

            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

            # image is (256 x 256 x input_c_dim)
            e1 = conv2d(add_noise(image), self.aef_dim, name='ae_e1_conv')
            # e1 is (128 x 128 x self.aef_dim)
            e2 = self.ae_bn_e2(conv2d(lrelu(add_noise(e1,self.noise_std_dev)), self.aef_dim*2, name='ae_e2_conv'))
            # e2 is (64 x 64 x self.aef_dim*2)
            e3 = self.ae_bn_e3(conv2d(lrelu(add_noise(e2,self.noise_std_dev)), self.aef_dim*4, name='ae_e3_conv'))
            # e3 is (32 x 32 x self.aef_dim*4)
            e4 = self.ae_bn_e4(conv2d(lrelu(add_noise(e3,self.noise_std_dev)), self.aef_dim*8, name='ae_e4_conv'))
            # e4 is (16 x 16 x self.aef_dim*8)
            e5 = self.ae_bn_e5(conv2d(lrelu(add_noise(e4,self.noise_std_dev)), self.aef_dim*8, name='ae_e5_conv'))
            # e5 is (8 x 8 x self.aef_dim*8)
            e6 = self.ae_bn_e6(conv2d(lrelu(add_noise(e5,self.noise_std_dev)), self.aef_dim*8, name='ae_e6_conv'))
            # e6 is (4 x 4 x self.aef_dim*8)
            e7 = self.ae_bn_e7(conv2d(lrelu(add_noise(e6,self.noise_std_dev)), self.aef_dim*8, name='ae_e7_conv'))
            # e7 is (2 x 2 x self.aef_dim*8)
            e8 = self.ae_bn_e8(conv2d(lrelu(add_noise(e7,self.noise_std_dev)), self.aef_dim*8, name='ae_e8_conv'))
            # e8 is (1 x 1 x self.aef_dim*8)

            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(add_noise(e8,self.noise_std_dev)),
                [self.batch_size, s128, s128, self.aef_dim*8], name='ae_d1', with_w=True)
            d1 = tf.nn.dropout(self.ae_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e7], 3)
            # d1 is (2 x 2 x self.aef_dim*8*2)

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(add_noise(d1,self.noise_std_dev)),
                [self.batch_size, s64, s64, self.aef_dim*8], name='ae_d2', with_w=True)
            d2 = tf.nn.dropout(self.ae_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e6], 3)
            # d2 is (4 x 4 x self.aef_dim*8*2)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(add_noise(d2,self.noise_std_dev)),
                [self.batch_size, s32, s32, self.aef_dim*8], name='ae_d3', with_w=True)
            d3 = tf.nn.dropout(self.ae_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e5], 3)
            # d3 is (8 x 8 x self.aef_dim*8*2)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(add_noise(d3,self.noise_std_dev)),
                [self.batch_size, s16, s16, self.aef_dim*8], name='ae_d4', with_w=True)
            d4 = self.ae_bn_d4(self.d4)
            d4 = tf.concat([d4, e4], 3)
            # d4 is (16 x 16 x self.aef_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(add_noise(d4,self.noise_std_dev)),
                [self.batch_size, s8, s8, self.aef_dim*4], name='ae_d5', with_w=True)
            d5 = self.ae_bn_d5(self.d5)
            d5 = tf.concat([d5, e3], 3)
            # d5 is (32 x 32 x self.aef_dim*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(add_noise(d5,self.noise_std_dev)),
                [self.batch_size, s4, s4, self.aef_dim*2], name='ae_d6', with_w=True)
            d6 = self.ae_bn_d6(self.d6)
            d6 = tf.concat([d6, e2], 3)
            # d6 is (64 x 64 x self.aef_dim*2*2)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(add_noise(d6,self.noise_std_dev)),
                [self.batch_size, s2, s2, self.aef_dim], name='ae_d7', with_w=True)
            d7 = self.ae_bn_d7(self.d7)
            d7 = tf.concat([d7, e1], 3)
            # d7 is (128 x 128 x self.aef_dim*1*2)

            self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(add_noise(d7,self.noise_std_dev)),
                [self.batch_size, s, s, self.output_c_dim], name='ae_d8', with_w=True)
            # d8 is (256 x 256 x output_c_dim)

            return tf.nn.tanh(self.d8)
    def save(self, checkpoint_dir, step):
        model_name = "pix2pix.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(self.checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(self.checkpoint_dir, model_name),
                        global_step=step)
    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        self.saver.restore(self.sess, checkpoint)
        return True
