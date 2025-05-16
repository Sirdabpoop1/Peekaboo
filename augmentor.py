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

for partition in ['train', 'test', 'val']:
    for image in os.listdir(os.path.join('data', partition, 'images')):
        img = cv2.imread(os.path.join('data', partition, 'images', image))
        
        coords = [0, 0, 0.00001, 0.00001]
        label_path = os.path.join('data', partition, 'labels', f'{image.split('.')[0]}.json')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label = json.load(f)

            coords[0] = label['shapes'][0]['points'][0][0]
            coords[1] = label['shapes'][0]['points'][0][1]
            coords[2] = label['shapes'][0]['points'][1][0]
            coords[3] = label['shapes'][0]['points'][1][1]
            coords = list(np.divide(coords, [640, 480, 640, 480]))

        try:
            for x in range(60):
                augmented = augmentor(image = img, bboxes = [coords], class_labels = ['face'])
                cv2.imwrite(os.path.join('aug'))
                cv2.rectangle(augmented['image'],




img = cv2.imread(os.path.join('data', 'train', 'images', '8a1c45f7-26d5-11f0-8723-b81ea44562d0.jpg'))

with open(os.path.join('data', 'train', 'labels', '8a1c45f7-26d5-11f0-8723-b81ea44562d0.json'), 'r') as f:
    label = json.load(f)
    coords = [0, 0, 0, 0]
    coords[0] = label['shapes'][0]['points'][0][0]
    coords[1] = label['shapes'][0]['points'][0][1]
    coords[2] = label['shapes'][0]['points'][1][0]
    coords[3] = label['shapes'][0]['points'][1][1]
    coords = list(np.divide(coords, [640, 480, 640, 480]))
    augmented = augmentor(image = img, bboxes = [coords], class_labels = ['face'])
    cv2.rectangle(augmented['image'],
                  tuple(np.multiply(augmented['bboxes'][0][:2], [450, 450]).astype(int)),
                  tuple(np.multiply(augmented['bboxes'][0][2:], [450, 450]).astype(int)),
                  (255, 0, 0), 2)

plt.imshow(augmented['image'])

plt.show()