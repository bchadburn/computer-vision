import tensorflow as tf
import numpy as np


def parse_image_to_tensor(filename, image_size):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [image_size, image_size])
    return image


def process_single_image(image_path, image_size):
    image_tensor = parse_image_to_tensor(image_path, image_size)
    x = np.expand_dims(image_tensor, axis=0)
    image = np.vstack([x])
    return image
