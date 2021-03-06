import tensorflow as tf
import numpy as np
import pathlib

data_root = pathlib.Path('../input/train_data/')

import random

all_image_paths = list(data_root.glob('./*'))
all_image_paths = [str(path) for path in all_image_paths]

AUTOTUNE = tf.data.experimental.AUTOTUNE

import os
import IPython.display as display
import random

def caption_image(image_path):
    if 'ok' in image_path:
        return "OK"
    else:
        return "DEF"

for n in range(3):
    image_path = random.choice(all_image_paths)
    display.display(display.Image(image_path))
    print(caption_image(image_path))


label_names = ['ok', 'def']

label_to_index = dict((name, index) for index, name in enumerate(label_names))


def mk_idx(x):
    if 'ok' in x:
        return 'ok'
    else:
        return 'def'

all_image_labels = [label_to_index[mk_idx(path)] for path in all_image_paths]

img_path =all_image_paths[0]

img_raw = tf.io.read_file(image_path)
img_tensor = tf.image.decode_image(img_raw)

img_final = tf.image.resize(img_tensor, [192,192])
img_final = img_final/255.0
print(img_final.shape)
print(img_final.numpy().min())
print(img_final.numpy().max())

def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192,192])
    image /= 255.0

    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)

image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

plt.figure(figsize=(8,8))

for n, image in enumerate(image_ds.take(4)):
    plt.subplot(2,2,n+1)
    plt.imshow(image)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(caption_image(all_image_paths[n]))
    plt.show()

label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))

for label in label_ds.take(4):
    print(label_names[label.numpy()])

image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

BATCH_SIZE = 32
image_count = len(all_image_paths)

ds = image_label_ds.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)

ds = ds.prefetch(buffer_size=AUTOTUNE)

mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192,192,3), include_top=False)
mobile_net.trainable = False

def change_range(image, label):
    return 2*image -1, label

keras_ds = ds.map(change_range)

image_batch, label_batch = next(iter(keras_ds))

feature_map_batch = mobile_net(image_batch)

model = tf.keras.Sequential([
    mobile_net,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(len(label_names))
])


logit_batch = model(image_batch).numpy()

model.compile(optimizer=tf.keras.optimizers.Adam(),loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

steps_per_epoch = tf.math.ceil(len(all_image_paths)/BATCH_SIZE).numpy()
