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

def caption_image(image_path):
    if 'ok' in image_path:
        return "OK"
    else:
        return "DEF"

for n in range(3):
    image_path = random.choice(all_image_paths)
    display.display(display.Image(image_path))
    print(caption_image(image_path))
