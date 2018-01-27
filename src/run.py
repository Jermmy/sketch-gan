from util import DataGenerator
from model import SimpleGan
import tensorflow as tf
import numpy as np

from config import *

saving_model = True
loading_model = False

if __name__ == '__main__':
    batch_size = 10
    epoch = 5
    noise_dim = 1000

    generator = DataGenerator(batch_size=batch_size, epoch=epoch)
    iterator = generator.train_dataset.make_one_shot_iterator()

    image, noise_input = iterator.get_next()

    gan = SimpleGan()

    gan_loss, disc_loss = gan.train(noise_input, image)

    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.0001
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100, 0.8, staircase=True)
    train_gan = tf.contrib.layers.optimize_loss(
        loss=gan_loss,
        global_step=global_step,
        learning_rate=learning_rate,
        optimizer=tf.train.AdamOptimizer,
        clip_gradients=9.0,
        summaries=["learning_rate", "loss"]
    )
    train_disc = tf.contrib.layers.optimize_loss(
        loss=disc_loss,
        global_step=global_step,
        learning_rate=learning_rate,
        optimizer=tf.train.AdamOptimizer,
        summaries=["learning_rate", "loss"]
    )

    with tf.Session() as sess:
        saver = tf.train.Saver()
        if loading_model:
            saver.restore(sess, model_file)
        else:
            tf.global_variables_initializer().run()

        try:
            step = 0
            while True:
                _, gl = sess.run([train_gan, gan_loss])
                _, dl = sess.run([train_disc, disc_loss])

                if step % 10 == 0:
                    print("Minibatch at step %d ==== gan_loss: %.2f, disc_loss: %.2f" % (step, gl, dl))
                if (step+1) % 100 == 0 and saving_model:
                    save_path = saver.save(sess, model_file)
                    print("Model saved in %s" % save_path)

                step += 1
        except tf.errors.OutOfRangeError:
            if saving_model:
                save_path = saver.save(sess, model_file)
                print("Model saved in %s" % save_path)



