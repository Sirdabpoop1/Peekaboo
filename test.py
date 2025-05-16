import cv2
import os
import time
import numpy as np
import uuid
import json
from matplotlib import pyplot as plt
import albumentations as alb

augmentor = alb.Compose([alb.RandomCrop(width = 450, height = 450),
                         alb.HorizontalFlip(p = 0.5),
                         alb.RandomBrightnessContrast(p = 0.2),
                         alb.RandomGamma(p = 0.2),
                         alb.RGBShift(p = 0.2),
                         alb.VerticalFlip(p = 0.2)],
                         bbox_params = alb.BboxParams(format = 'albumentations',
                                                      label_fields = ['class_labels']))

cv2.iamread(os.path.join("data"))