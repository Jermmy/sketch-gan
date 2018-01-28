from os.path import isdir, exists
import cairosvg
import cv2
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from config import *


def render_svg_file():
    svg_dirs = [join(svg_data_dir, d) for d in os.listdir(svg_data_dir)
                if isdir(join(svg_data_dir, d))]

    svg_dirs = svg_dirs[0:30]

    for d in svg_dirs:

        dest_dir = join(sketch_data_dir, d.split('/')[-1])

        if exists(dest_dir):
            continue

        os.mkdir(dest_dir)

        svg_files = [join(d, f) for f in os.listdir(d) if f.endswith(".svg")]
        for f in svg_files:
            new_file = join(dest_dir, f.split('/')[-1].split('.')[0] + ".png")
            cairosvg.svg2png(url=f, write_to=new_file)

            print("render to %s" % (new_file))
            image = cv2.imread(new_file, cv2.IMREAD_UNCHANGED)

            b, g, r, a = cv2.split(image)
            image = 255 - a
            image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
            cv2.imwrite(new_file, image)


class DataGenerator():

    def __init__(self, noise_dim=1000, batch_size=10, epoch=5):
        train_files = self._get_all_train_files()
        self.noise_dim = noise_dim
        self.batch_size = batch_size

        self.train_dataset = tf.data.Dataset.from_tensor_slices((train_files))
        self.train_dataset = self.train_dataset.map(self._parse_data)
        # self.train_dataset = self.train_dataset.map(
        #     lambda file: tuple(tf.py_func(
        #         self._parse_data, [file], [tf.float32, tf.float32]
        #     ))
        # )
        self.train_dataset = self.train_dataset.batch(batch_size).repeat(epoch)


    def _get_all_train_files(self):
        train_files = []
        dirs = [join(sketch_data_dir, d) for d in os.listdir(sketch_data_dir)
                if isdir(join(sketch_data_dir, d))]
        for d in dirs:
            files = [join(d, f) for f in os.listdir(d) if f.endswith(".png")]
            train_files.extend(files)

        train_files = shuffle(train_files)

        return train_files

    def _get_one_train_file(self):
        train_files = []
        dirs = [join(sketch_data_dir, d) for d in os.listdir(sketch_data_dir)
                if isdir(join(sketch_data_dir, d))]
        # for d in dirs:
        d = dirs[0]
        files = [join(d, f) for f in os.listdir(d) if f.endswith(".png")]
        train_files.extend(files)

        train_files = shuffle(train_files)
        return train_files

    def _parse_data(self, filename):
        image_string = tf.read_file(filename=filename)
        image_decode = tf.image.decode_image(image_string)
        image_decode = tf.cast(image_decode, tf.float32)
        image_decode = tf.subtract(image_decode, 255.0 / 2)
        image_decode = image_decode / 255.0

        noise_input = np.random.uniform(-1., 1., size=[self.noise_dim]).astype(np.float32)

        return image_decode, noise_input

    # def _parse_data(self, filename):
    #     image = cv2.imread(filename.decode(), cv2.IMREAD_UNCHANGED)
    #     image = image.reshape(image_size, image_size, 1).astype(np.float32)
    #     image = (image - 255.0 / 2) / 255.0
    #
    #     # noise_input = tf.random_uniform(shape=[self.noise_dim,],
    #     #                                 minval=-1.0, maxval=1.0, dtype=tf.float32)
    #
    #     noise_input = np.random.uniform(-1., 1., size=[self.noise_dim]).astype(np.float32)
    #
    #     return image, noise_input



if __name__ == '__main__':
    render_svg_file()
    # generator = DataGenerator()
    # iterator = generator.train_dataset.make_one_shot_iterator()
    # image, noise_input = iterator.get_next()
    # print(image)
    # print(noise_input)
    #
    # with tf.Session() as sess:
    #     try:
    #         while True:
    #             i, n = sess.run([image, noise_input])
    #             # print(e[0])
    #             # print(e[1])
    #             print(i.shape)
    #             print(n.shape)
    #     except tf.errors.OutOfRangeError:
    #         print("end")
