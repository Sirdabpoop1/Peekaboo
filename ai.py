#Dependencies
import cv2
import os
import time
import numpy as np
import uuid
import json
from matplotlib import pyplot as plt
import albumentations as alb

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf


#Avoid out of memory errors by setting GPU growth
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

tf.config.list_physical_devices("GPU")

images = tf.data.Dataset.list_files('data\\images\\*.jpg')

def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img

images = images.map(load_image)

image_gen = images.batch(4).as_numpy_iterator()
plot_images = image_gen.next()

fig, ax = plt.subplots(ncols = 4, figsize = (20, 20))
for idx, image in enumerate(plot_images):
    ax[idx].imshow(image)
plt.show()