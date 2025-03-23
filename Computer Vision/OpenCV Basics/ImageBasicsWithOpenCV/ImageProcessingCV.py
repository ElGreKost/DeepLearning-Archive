import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# Open cv and matplotlib expects different order in colors
# Matplotlib --> RGB || OpenCV --> BGR

img = cv.imread('../DATA/00-puppy.jpg')

"""
# Color conversion for open ccv from rgb to bgr
fix_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# for grayscale matplotlib usesanother colormapping for dark and light
img_gray = cv.imread('../DATA/00-puppy.jpg', cv.IMREAD_GRAYSCALE)

# print(img_gray.shape)

# To resize image:
# print(fix_img.shape)
# by pixel
pix_resz_img = cv.resize(fix_img, (1000, 400))
w_ratio, h_ratio = 0.5, 0.5
# by ratio
rat_resz_img = cv.resize(fix_img, (0, 0), fix_img, w_ratio, h_ratio)

# flip image: {-1 -> vertical | 1 -> horizontal
#               0 -> both }
flipped = cv.flip(fix_img, 1)

# save image file:
cv.imwrite('output.jpg', fix_img)

# manage display window (not useful for pycharm)
figure = plt.figure(figsize=(10,8))
axes = figure.add_subplot(111)
axes.imshow(fix_img)

plt.show()
"""

#############################################
"""
while True:
    cv.imshow('Puppy', img)
    # if we waited at least 1 milisec
    # and we've pressed the Esc key
    if cv.waitKey(1) & 0xFF == 27:
        break
cv.destroyAllWindows()
"""


############################################

def plot(img):
    """my func for plotting """
    plt.imshow(img)
    plt.show()


"""
blank_img = np.zeros(shape=(512, 512, 3), dtype=np.int16)

# make rectangle
cv.rectangle(blank_img, pt1=(384, 0), pt2=(510, 150), color=(0, 255, 0), thickness=10)
plot(blank_img)

# make square
cv.rectangle(blank_img, pt1=(200, 200), pt2=(300, 300), color=(0, 0, 255), thickness=10)
plot(blank_img)

# make circle
cv.circle(img=blank_img, center=(100, 100), radius=50, color=(255, 0, 0), thickness=8)
plot(blank_img)

# to fill it in make thickness -1
cv.circle(img=blank_img, center=(400, 400), radius=50, color=(255, 0, 0), thickness=-1)
plot(blank_img)

# draw line
cv.line(blank_img, pt1=(0, 0), pt2=(512, 512), color=(102, 255, 255), thickness=5)
plot(blank_img)

#########################################

# Write on image
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(blank_img,text="Hello", org=(10,500), fontFace=font,fontScale=4,color=(255,255,255),
           thickness=3,lineType=cv.LINE_AA)
plot(blank_img)

# Custom polygon
blank_img = np.zeros(shape=(512,512,3),dtype=np.int32)
vertices = np.array([ [100,300], [200,200], [400,300], [200,400] ], dtype=np.int32)
# now vertices have to be in 3d so we reshape by adding 1 in the 2nd dimension
pts = vertices.reshape(-1,1,2)
print(vertices.shape)
print(pts.shape)

# pass points as list
cv.polylines(blank_img,[pts], isClosed=True, color=(255,0,0),thickness=5)
plot(blank_img)"""


#############################################

## Draw with mouse on image ##

## FUNCTION ##

def draw_circle(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(img, (x, y), 100, (0, 255, 0), -1)

    elif event is cv.EVENT_RBUTTONDOWN:
        cv.circle(img, (x, y), 100, (255, 0, 0), -1)


# variable for draw_rectangle
# True while mouse button down, False while mouse button Up
drawing = False
ix, iy = -1, -1


def draw_rectangle(event, x, y, flags, params):
    """ix, iy are the starting coordinates for the rectangle"""
    global ix, iy, drawing

    if event is cv.EVENT_LBUTTONDOWN:   # START
        drawing = True
        ix,iy = x,y

    elif event is cv.EVENT_MOUSEMOVE:   # DRAW
        if drawing:
            cv.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)

    elif event is cv.EVENT_LBUTTONUP:   # STOP
        drawing = False
        cv.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)


"""cv.namedWindow(winname="my_drawing")

cv.setMouseCallback('my_drawing', draw_circle)

# if i do not use np.int8 - image get totally black
img = np.zeros((512, 512, 3), np.int8)

while True:
    cv.imshow('my_drawing', img)

    if cv.waitKey(20) & 0xFF == 27:
        break
"""

img = np.zeros((512, 512, 3))

cv.namedWindow(winname='my_drawing')

cv.setMouseCallback('my_drawing', draw_rectangle)

while True:
    cv.imshow('my_drawing', img)

    if cv.waitKey(1) & 0xFF == 27:
        break

cv.destroyAllWindows()
