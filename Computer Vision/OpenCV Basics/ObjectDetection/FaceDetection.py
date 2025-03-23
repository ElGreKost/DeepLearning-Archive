import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

nadia = cv.imread('../DATA/Nadia_Murad.jpg', 0)
denis = cv.imread('../DATA/Denis_Mukwege.jpg', 0)
solvay = cv.imread('../DATA/solvay_conference.jpg', 0)

# use pretrained xml file ( classifier)
face_cascade = cv.CascadeClassifier('../DATA/haarcascades/haarcascade_frontalface_default.xml')


def detect_face(img):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img)
    for (x, y, w, h) in face_rects:
        cv.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 10)

    return face_img


def adj_detect_face(img):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5)
    for (x, y, w, h) in face_rects:
        cv.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 10)

    return face_img


result = adj_detect_face(solvay)

# plt.imshow(result, cmap='gray')
# plt.show()


# not working that well with dark photos
# working on nadia not on denis or solvay
eye_cascade = cv.CascadeClassifier('../DATA/haarcascades/haarcascade_eye.xml')


def detect_eye(img):
    face_img = img.copy()
    eyes_rects = eye_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5)
    for (x, y, w, h) in eyes_rects:
        cv.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 10)

    return face_img


result = detect_eye(nadia)
plt.imshow(result, cmap='gray')
plt.show()

cap = cv.VideoCapture(0)

while True:
    ret,frame = cap.read(0)
    frame = detect_face(frame)
    cv.imshow('Video Face Detect',frame)
    k = cv.waitKey(1)
    if k == 27: break
cap.release()
cv.destroyAllWindows()

