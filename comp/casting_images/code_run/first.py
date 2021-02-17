import tensorflow as tf
import numpy as np
import pathlib

data_root = pathlib.Path('../input/train_data/')

import random

all_image_paths = list(data_root.glob('./*'))
all_image_paths = [str(path) for path in all_image_paths]

import os
import IPython.display as display
import random



label_names = ['ok', 'def']

label_to_index = dict((name, index) for index, name in enumerate(label_names))


def mk_idx(x):
    if 'ok' in x:
        return 'ok'
    else:
        return 'def'

all_image_labels = [label_to_index[mk_idx(path)] for path in all_image_paths]

img_path =all_image_paths[0]

img_raw = tf.io.read_file(img_path)
img_tensor = tf.image.decode_image(img_raw)

img_final = tf.image.resize(img_tensor, [192,192])
img_final = img_final/255.0
print(img_final.shape)
print(img_final.numpy().min())
print(img_final.numpy().max())

#def preprocess_image(image):
#    image = tf.image.decode_jpeg(image, channels=3)
#    image = tf.image.resize(image, [192,192])
#    image /= 255.0
#
#    return image
#
#def load_and_preprocess_image(path):
#    image = tf.io.read_file(path)
#    return preprocess_image(image)
