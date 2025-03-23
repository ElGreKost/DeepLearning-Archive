import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def plot(img, title='', cmap='gray'):
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.show()


# sep_coins = cv.imread('../DATA/pennies.jpg')
# plot(sep_coins, 'coins')

""" TRY NON WATERSHED APPROACH {ENDS IN FAILING} """
"""# Median Blur ( To remove useless info )
sep_blur = cv.medianBlur(sep_coins, 25)
# Convert to Grayscale
gray_sep_coins = cv.cvtColor(sep_blur, cv.COLOR_BGR2GRAY)
plot(gray_sep_coins, 'gray coins')
# Binary Threshold ( to make it black and white)
ret, sep_thresh = cv.threshold(gray_sep_coins, 160, 255, cv.THRESH_BINARY_INV)
plot(sep_thresh, 'binary coins')
# Find Contours
contours, hierarchy = cv.findContours(sep_thresh.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:  # if external
        cv.drawContours(sep_coins, contours, i, (255, 0, 0), 10)
plot(sep_coins, 'False contour draw')"""

""" USING WATERSHED """
"""img = sep_coins
# High analysis in pic => we use high kernel size
img = cv.medianBlur(img, 35)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# if original threshold is used then some details remain
# So we use Otsu's method for more accurate result
# ret, thresh = cv.threshold(gray,127,255,cv.THRESH_BINARY_INV) #NOT
ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
plot(thresh)
# now use noise removal (optional- because image is very simple)
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
plot(opening)

# Grab what we are sure the background is
sure_bg = cv.dilate(opening, kernel, iterations=3)
plot(sure_bg, "sure bg")

## DISTANCE TRANSFORM ## ( find centers )
# The further away you go from black the brighter you become
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
plot(dist_transform)

# keep the middle of coins
ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
plot(sure_fg)

# now we use the watershed algo to find the white required around
#   each middle part of coins that is now unknown due to thresholding.
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)
plot(unknown)

# create the label markers for watershed algo
# 1. get markers
ret, markers = cv.connectedComponents(sure_fg)
plot(markers, 'fit centers as components || turn them to markers')
markers += 1  # make background kind of gray
plot(markers, 'background turn to gray')
markers[unknown == 255] = 0 # make the sure_bg area total black
plot(markers, 'sure background (coin"s area around center) is now black')

# 3. use watershed
markers = cv.watershed(img, markers)
plot(markers, 'after using watershed ')

# Find Contours
contours, hierarchy = cv.findContours(markers.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:  # if external
        cv.drawContours(sep_coins, contours, i, (255, 0, 0), 10)
plot(sep_coins, 'Result from watershed')
"""

""" USING WATERSHED WITH MANUALLY PUT MARKERS"""
road = cv.imread('../DATA/road_image.jpg')
road_copy = np.copy(road)

marker_image = np.zeros(road.shape[:2], dtype=np.int32)
segments = np.zeros(road.shape, dtype=np.uint8)

from matplotlib import cm  # color mapping

cm.tab10(0)  # they are valued from 0-1 so make them go to 255


def create_rgb(i):
    tuple(np.array(cm.tab10(i)[:3] * 255))


colors = []
for i in range(10):
    colors.append(create_rgb(i))

###
# GLOBAL VARIABLES
n_markers = 10 # 0-9
current_marker = 1  # COLOR CHOICE
marks_updated = False  # MARKERS UPDATED BY WATERSHED
# CALLBACK FUNCTION
def mouse_callback(event,x,y,flags,param):
    global marks_updated

    if event is cv.EVENT_LBUTTONDOWN:
        # MARKERS PASSED TO THE WATESHED ALGO
        cv.circle(marker_image,(x,y),10,(current_marker),-1)

        # USER SEES ON THE ROAD IMAGE
        cv.circle(road_copy,(x,y),10,colors[current_marker],-1)
        marks_updated = True

# WHILE TRUE
cv.namedWindow('Road Image')
cv.setMouseCallback('Road Image',mouse_callback)

while 1:
    cv.imshow('Watershed Segments', segments)
    cv.imshow('Road Image',road_copy)

    # CLOSE ALL WINDOWS
    k = cv.waitKey(1)

    if k == 27: break

    # CLEARING ALL COLORS on PRESS C KEY
    elif k == ord('c'):
        road_copy = road.copy()
        marker_image = np.zeros(road.shape[:2], dtype=np.int32)
        segments = np.zeros(road.shape, dtype=np.uint8)

    # UPDATE COLOR CHOICE
    elif k > 0 and chr(k).isdigit():
        current_marker = int(chr(k))

    # UPDATE THE MARKINGS
    if marks_updated:

        marker_image_copy = marker_image.copy()
        cv.watershed(road,marker_image_copy)

        segments = np.zeros(road.shape,dtype=np.uint8)

        # TODO  The error is here probably
        for clr_ind in range(n_markers):
            # COLORING SEGMENTS , NUMPY CALL
            segments[marker_image_copy == clr_ind] = colors[clr_ind]

cv.destroyAllWindows()
