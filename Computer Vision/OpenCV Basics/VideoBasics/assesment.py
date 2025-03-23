import cv2 as cv
import matplotlib.pyplot as plt

cap = cv.VideoCapture(0)


def draw_circle(event, x, y, flags, param):
    global center, clicked

    if event is cv.EVENT_LBUTTONDOWN:
        center = (x,y)
        clicked = False
    elif event is cv.EVENT_LBUTTONUP:
        clicked = True

center, clicked = (0, 0), False

cv.namedWindow('frame')
cv.setMouseCallback('frame', draw_circle)


while 1:
    ret, frame = cap.read()

    if clicked:
        cv.circle(frame,center=center, radius=(30), thickness=(5), color=(255,0,0))


    cv.imshow('frame', frame)

    if cv.waitKey(5) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
