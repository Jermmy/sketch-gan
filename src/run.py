from util import DataGenerator
from model import SimpleGan
import tensorflow as tf
import numpy as np
import cv2
from config import *

saving_model = True
loading_model = False

batch_size = 10
epoch = 5
noise_dim = 1000


def train():
    generator = DataGenerator(batch_size=batch_size, epoch=epoch)
    iterator = generator.train_dataset.make_one_shot_iterator()

    image, noise_input = iterator.get_next()

    gan = SimpleGan()

    gen_loss, disc_loss = gan.train(noise_input, image)

    global_step = tf.Variable(0, trainable=False)
    gen_start_learning_rate = 0.000001
    gen_learning_rate = tf.train.exponential_decay(gen_start_learning_rate, global_step, 100, 0.8, staircase=True)
    disc_start_learning_rate = 0.0001
    disc_learning_rate = tf.train.exponential_decay(disc_start_learning_rate, global_step, 100, 0.8, staircase=True)
    train_gan = tf.contrib.layers.optimize_loss(
        loss=gen_loss,
        global_step=global_step,
        learning_rate=gen_learning_rate,
        optimizer=tf.train.AdamOptimizer,
        clip_gradients=9.0,
        summaries=["learning_rate", "loss"]
    )
    train_disc = tf.contrib.layers.optimize_loss(
        loss=disc_loss,
        global_step=global_step,
        learning_rate=disc_learning_rate,
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
                # _, gl = sess.run([train_gan, gen_loss])
                _, dl = sess.run([train_disc, disc_loss])

                for i in range(10):
                    # _, dl = sess.run([train_disc, disc_loss])
                    _, gl = sess.run([train_gan, gen_loss])

                _, dl = sess.run([train_disc, disc_loss])

                if step % 10 == 0:
                    print("Minibatch at step %d ==== gen_loss: %.2f, disc_loss: %.2f" % (step, gl, dl))
                    # print("Minibatch at step %d ==== disc_loss: %.2f" % (step, dl))
                if (step+1) % 100 == 0 and saving_model:
                    save_path = saver.save(sess, model_file)
                    print("Model saved in %s" % save_path)

                step += 1
        except tf.errors.OutOfRangeError:
            if saving_model:
                save_path = saver.save(sess, model_file)
                print("Model saved in %s" % save_path)



def generate():
    noise_input = tf.placeholder(tf.float32, [1, noise_dim])
    gan = SimpleGan()
    result = gan.gen_sample(noise_input)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_file)

        input = np.random.uniform(-1., 1., size=[1, noise_dim])
        image = sess.run(result, feed_dict={noise_input: input})

        image = image.reshape((image_size, image_size, image_channel))
        print(image.shape)

        image = image * 255.0 + 127.0



        cv2.imwrite(gen_sample_dir + "test.png", image)





if __name__ == '__main__':
    train()
    # generate()





