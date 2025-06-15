import os
import time
import uuid
import cv2

IMAGES_PATH = os.path.join('data', 'val', 'labels')
number_images = 13

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    
    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)
    if key == ord('w'):
        for imgnum in range(number_images):
            print('Collecting image {}'.format(imgnum))
            ret, frame = cap.read()
            imgname = os.path.join(IMAGES_PATH,f'{str(uuid.uuid1())}.jpg')
            cv2.imwrite(imgname, frame)
            cv2.imshow('frame', frame)
            cv2.waitKey(500)
    elif key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()