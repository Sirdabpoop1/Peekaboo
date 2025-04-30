#Dependencies
import cv2
import os
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









#Setup paths
POS_PATH = os.path.join("data", "positive")
NEG_PATH = os.path.join("data", "negative")
ANC_PATH = os.path.join("data", "anchor")

#Making folders
os.makedirs(POS_PATH, exist_ok=True)
os.makedirs(NEG_PATH, exist_ok=True)
os.makedirs(ANC_PATH, exist_ok=True)

#Sets up camera
cam = cv2.VideoCapture(0)
while cam.isOpened():
    ret, frame = cam.read()

    #Cut down the frame to 250px x 250px
    frame = frame[120:120+250, 200:200+250, :]

    #Take anchors
    if cv2.waitKey(1) & 0XFF == ord("a"):
        imgname = os.path.join(ANC_PATH, "{}.jpg".format(uuid.uuid1()))
        cv2.imwrite(imgname, frame)
    #Take positives
    if cv2.waitKey(1) & 0XFF == ord("p"):
        imgname = os.path.join(POS_PATH, "{}.jpg".format(uuid.uuid1()))
        cv2.imwrite(imgname, frame)

    cv2.imshow("Camera", frame)
    
    if cv2.waitKey(1) & 0XFF == ord("q"):
        break

#Releases the webcam
cam.release()
#Closes the video
cv2.destroyAllWindows()
