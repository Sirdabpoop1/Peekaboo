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

train_images = tf.data.Dataset.list_files('aug_data\\train\\images\\*.jpg', shuffle = False)
train_images = train_images.map(load_image)
train_images = train_images.map(lambda x: tf.image.resize(x, (120, 120)))
train_images = train_images.map(lambda x: x/225)

test_images = tf.data.Dataset.list_files('aug_data\\test\\images\\*.jpg', shuffle = False)
test_images = train_images.map(load_image)
test_images = train_images.map(lambda x: tf.image.resize(x, (120, 120)))
test_images = train_images.map(lambda x: x/225)

val_images = tf.data.Dataset.list_files('aug_data\\val\\images\\*.jpg', shuffle = False)
val_images = train_images.map(load_image)
val_images = train_images.map(lambda x: tf.image.resize(x, (120, 120)))
val_images = train_images.map(lambda x: x/225)
