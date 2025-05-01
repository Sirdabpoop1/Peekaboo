#Dependencies
import cv2
import os
import time
import random
import numpy as np
import uuid
from matplotlib import pyplot as plt


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

#Import Tensorflow (deep learning) dependencies
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model

#Avoid out of memory errors by setting GPU growth
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    print(gpu)
    tf.config.experimental.set_memory_growth(gpu, True)

IMAGES_PATH = os.path.join("data", "images")
number_images = 30

cam = cv2.VideoCapture(0)
for imgnum in range(number_images):
    print("Collecting Image #{}".format(imgnum))
    ret, frame = cam.read()
    imgname = os.path.join(IMAGES_PATH, f"{str(uuid.uuid1())}.jpg")
    cv2.imwrite(imgname, frame)
    cv2.imshow("frame", frame)
    time.sleep(0.5)

    if cv2.waitKey(1) & 0XFF == ord('q'):
        break



cv2.release()
cv2.destroyAllWindows()