import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

image = cv2.imread('rowing.jpg')

#convert to grey
image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


#Training file
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_detect = face_cascade.detectMultiScale(image_grey, scaleFactor = 1.2, minNeighbors = 5);

print ('Faces found: ', len(face_detect))

for (x, y, w, h) in face_detect:
    cv2.rectangle(image, (x, y), (x+w, y+h), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 2)

retangle = np.ones((720, 1280))


cv2.imshow('rowing', image)





cv2.waitKey(100000)
cv2.destroyAllWindows()
