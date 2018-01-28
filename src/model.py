import tensorflow as tf

from config import *

class SimpleGan():

    def __init__(self):
        pass

    def train(self, gen_input, disc_input):
        return self._loss(gen_input, disc_input)

    def gen_sample(self, x):
        return self.generator(x)

    def generator(self, x, reuse=False):
        x = tf.reshape(x, shape=[-1, noise_dim])
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            x = tf.layers.dense(x, 9*9*64, name='fc')
            x = tf.nn.tanh(x)
            x = tf.reshape(x, shape=[-1, 9, 9, 64])

            dc1 = tf.layers.conv2d_transpose(x, strides=[3, 3], kernel_size=[6, 6],
                                             filters=32, padding='valid', name="dc1")
            dc2 = tf.layers.conv2d_transpose(dc1, strides=[3, 3], kernel_size=[13, 13],
                                             filters=1, padding='valid', name="dc2")

            # dc2 = tf.Print(dc2, [dc2], message="generator")

            dc2 = tf.nn.sigmoid(dc2)

            return dc2


    def discriminator(self, x, reuse=False):
        x = tf.reshape(x, shape=[-1, image_size, image_size, image_channel])
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            c1 = tf.layers.conv2d(x, filters=32, kernel_size=[13, 13], padding='VALID', name="conv1")
            c1 = tf.nn.relu(c1)
            c1 = tf.nn.max_pool(c1, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')

            c2 = tf.layers.conv2d(c1, filters=64, kernel_size=[7, 7], padding='VALID', name="conv2")
            c2 = tf.nn.relu(c2)
            c2 = tf.nn.max_pool(c2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

            c3 = tf.layers.conv2d(c2, filters=256, kernel_size=[3, 3], padding='VALID', name="conv3")
            c3 = tf.nn.relu(c3)
            c3 = tf.nn.max_pool(c3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

            shape = c3.get_shape().as_list()

            c3 = tf.reshape(c3, [-1, shape[1] * shape[2] * shape[3]])
            fc = tf.layers.dense(c3, 64, name='fc')

            # fc = tf.Print(fc, [fc], message="discriminator")

            fc = tf.nn.sigmoid(fc)
            return fc


    def _loss(self, gen_input, disc_input):
        gen_sample = self.generator(gen_input)

        disc_fake = self.discriminator(gen_sample)
        disc_real = self.discriminator(disc_input, reuse=True)

        gen_loss = -tf.reduce_mean(tf.log(disc_fake))
        disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))

        return gen_loss, disc_loss



