import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

flat_chess = cv.imread('../DATA/flat_chessboard.png')
flat_chess = cv.cvtColor(flat_chess, cv.COLOR_BGR2RGB)
# plt.imshow(flat_chess)
# plt.show()

gray_flat_chess = cv.cvtColor(flat_chess,cv.COLOR_BGR2GRAY)
# plt.imshow(gray_flat_chess, cmap='gray')
# plt.show()

real_chess = cv.imread('../DATA/real_chessboard.jpg')
real_chess = cv.cvtColor(real_chess,cv.COLOR_BGR2RGB)
# plt.imshow(real_chess)
# plt.show()

gray_real_chess = cv.cvtColor(real_chess,cv.COLOR_BGR2GRAY)
# plt.imshow(gray_real_chess, cmap='gray')
# plt.show()

# HARRIS CORNER DETECTION
"""## FLAT CHESS ##
# picture color values should be float
gray = np.float32(gray_flat_chess)

# ksize is the size of the kernel { corner algo uses kernel!!!socking)
dst = cv.cornerHarris(src=gray,blockSize=2,ksize=3,k=0.04)
dst = cv.dilate(dst,None)

flat_chess[dst>0.01*dst.max()] = [255,0,0] # RGB

plt.imshow(flat_chess)
plt.show()

## REAL CHESS
gray = np.float32(gray_real_chess)

dst = cv.cornerHarris(src=gray,blockSize=2,ksize=3,k=0.04)
dst = cv.dilate(dst,None)

real_chess[dst>0.01*dst.max()] = [255,0,0]
plt.imshow(real_chess)
plt.show()"""

"""# Shi-Tomasi Corner Detection
# returns a list of coordinats of the corners
corners = cv.goodFeaturesToTrack(gray_flat_chess,64,0.01,10)
corners = np.int0(corners)  # convert to int

for i in corners:
    x,y = i.ravel()  # flatten
    cv.circle(flat_chess,(x,y),3,(255,0,0),-1)

# gray_real_chess = cv.blur(gray_real_chess,ksize=(5,5))
# gray_real_chess = cv.GaussianBlur(gray_real_chess,(9,9),10)
# gray_real_chess = cv.medianBlur(gray_real_chess,9)


corners = cv.goodFeaturesToTrack(gray_real_chess,64,0.01,10)
corners = np.int0(corners)  # convert to int

for i in corners:
    x,y = i.ravel()  # flatten
    cv.circle(real_chess,(x,y),3,(255,0,0),-1)

plt.imshow(real_chess)
plt.show()"""

## EDGE DETECTION ##

img = cv.imread('../DATA/sammy_face.jpg')

med_val = np.median(img)

# set lower threshold to 0 or 70% of the median value whichever is greater
lower = int(max(0,0.7*med_val))
# set upper threshold 130% of median or 255 whichever is smaller
upper = int(min(255,1.3*med_val))

edges = cv.Canny(image=img, threshold1=lower,threshold2=upper+50)
# if the image is still bad try to change the lower and upper

blurred_img = cv.blur(img,ksize=(9,9))

edges = cv.Canny(image=blurred_img, threshold1=lower,threshold2=upper)


plt.imshow(edges)
plt.show()
