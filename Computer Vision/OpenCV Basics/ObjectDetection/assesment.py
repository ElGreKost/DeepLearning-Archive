import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def plot(img, title=''):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()


img = cv.imread('../DATA/car_plate.jpg')
# plot(img, 'cars')

plate_cascade = cv.CascadeClassifier('../DATA/haarcascades/haarcascade_russian_plate_number.xml')


def detect_plate(img):
    # scaleFactor=1.2, minNeighbors=5
    plate_coord = plate_cascade.detectMultiScale(img, scaleFactor=1.2)
    for x, y, w, h in plate_coord:
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 10)
    return img


def detect_and_blur_plate(img):
    # scaleFactor=1.2, minNeighbors=5
    plate_img = img.copy()
    roi = img.copy()
    plate_rects = plate_cascade.detectMultiScale(plate_img, scaleFactor=1.3,minNeighbors=3)
    print(plate_rects)
    for x, y, w, h in plate_rects:
        # make another picture as a copy of the previous one and blur it then put it where it should be

        roi = roi[y:y+h,x:x+w]
        roi = cv.medianBlur(roi,5)
        plate_img[y:y+h,x:x+w] = roi

    return plate_img


result = detect_and_blur_plate(img)
plot(result)
