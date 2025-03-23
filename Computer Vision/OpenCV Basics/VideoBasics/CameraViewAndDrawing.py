import cv2 as cv
# To show recorded video in the fps that it was recorded
import time

"""### GET VIDEO FROM CAMERA ###

capture = cv.VideoCapture(0)  # 0 for default camera

width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))

# *'DIVX' is a parameter used for windows there is a different for other os
writer = cv.VideoWriter('mysupervideo.mp4', cv.VideoWriter_fourcc(*'DIVX'), 20, (width,height))

while True:

    ret,frame =capture.read()

    # OPERATIONS (DRAWING)
    writer.write(frame)

    # gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    cv.imshow('frame', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
writer.release()
cv.destroyAllWindows()
"""

"""### Use existing video ###
cap = cv.VideoCapture('mysupervideo.mp4')

if cap.isOpened() is False:print("ERROR FILE NOT FOUND")

while cap.isOpened:

    ret,frame = cap.read()

    if ret: # is still returning frames

        # WRITER 20 FPS
        time.sleep(1/20)
        cv.imshow('frame', frame)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    else : break

cap.release()"""


### DRAW ON A VIDEO ###

## CALLBACK FUNCTION RECTANGLE
def draw_rectangle(event, x, y, flags, param):
    global pt1, pt2, topLeft_clicked, botRightClicked
    if event is cv.EVENT_LBUTTONDOWN:

        # RESET THE RECTANGLE
        if topLeft_clicked and botRightClicked:
            pt1 = pt2 = (0, 0)
            topLeft_clicked = botRightClicked = False

        if not topLeft_clicked:
            pt1 = (x,y)
            topLeft_clicked = True
        elif not botRightClicked:
            pt2 = (x,y)
            botRightClicked = True
## GLOBAL VARIABLES
pt1 = pt2 = (0, 0)
topLeft_clicked = botRightClicked = False

## CONNECT TO THE CALLBACK
cap = cv.VideoCapture(0)

cv.namedWindow('Test')
cv.setMouseCallback('Test', draw_rectangle)

"""## CONSTANT DRAWINGS ## 
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# POINTS OF RECTANGLE
# bottom right -> x + w, y + h
# // returns integer
x, y = width // 2, height // 2

# width and height of rectangle
w = width // 4
h = height // 4
"""

while 1:
    ret, frame = cap.read()

    # cv.rectangle(frame, (x,y), (x+w,y+h), color=(0,0,255),thickness=4) # FOR CONSTANT DRAWING

    # DRAWING ON THE FRAME BASED OF THE GLOBAL VARIABLES
    if topLeft_clicked:
        cv.circle(frame, center=pt1, radius=5, color=(0, 0, 255), thickness=-1)

    if topLeft_clicked & botRightClicked:
        cv.rectangle(frame, pt1, pt2, (0, 0, 255), 3)

    cv.imshow('Test', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
