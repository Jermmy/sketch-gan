import platform

from os.path import join, exists
import os

if "Darwin" in platform.system():
    root_dir = "/Users/xyz/Desktop/sketch_test/SHREC13/sketch_gan/"
elif "Linux" in platform.system():
    root_dir = "/home/liuwq/xyz/sketch-gan/"

if not exists(root_dir):
    os.mkdir(root_dir)

if "Darwin" in platform.system():
    svg_data_dir = "/Users/xyz/Desktop/sketch_test/SHREC13/svg/"
elif "Linux" in platform.system():
    svg_data_dir = "/home/liuwq/xyz/SHREC13/data/svg/"

sketch_data_dir = join(root_dir, "sketch/")

if not exists(sketch_data_dir):
    os.mkdir(sketch_data_dir)

image_size = 100
image_channel = 1
noise_dim = 1000

model_dir = root_dir + "tf_model"
if not exists(model_dir):
    os.mkdir(model_dir)

model_file = join(model_dir, "gan.ckpl")
