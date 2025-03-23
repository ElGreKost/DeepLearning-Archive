import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def display(img, cmap='gray'):
    plt.imshow(img)
    plt.show()


reeses = cv.imread('../DATA/reeses_puffs.png', 0)

cereals = cv.imread('../DATA/many_cereals.jpg', 0)

# WITH ORB DESCRIPTORS

orb = cv.ORB_create()

kp1, des1 = orb.detectAndCompute(reeses, None)

kp2, des2 = orb.detectAndCompute(cereals, None)

# bf = bruteforce
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
mathces = sorted(matches, key=lambda x: x.distance)

reeses_mathces = cv.drawMatches(reeses, kp1, cereals, kp2, mathces[:25], None, flags=2)
display(reeses_mathces)  # bad job



### NOT WORKING ###



# Now we will use sift descriptors good for different scale
sift = cv.xfeatures2d.SURF.create()
kp1,des1 = sift.detectAndCompute(reeses,None)
kp2,des2 = sift.detectAndCompute(cereals,None)

bf = cv.BFMatcher()

# finds the k best matches for each descriptor
mathces = bf.knnMatch(des1, des2,k=2)

# now apply a ratio test
# if first match very close to second match then it is a good feature to match on
good = []

# LESS DISTANCE MEANS BETTER MATCH
# RATIO TEST
for match1,match2 in matches:
    if match1.distance < 0.75*match2.distance:
        good.append([match1])

sift_matches = cv.drawMatchesKnn(reeses,kp1,cereals,kp2,good,None,flags=2)
display(sift_matches)

# FLANN
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params,search_params)

matches = flann.KnnMatch(des1,des2,k=2)

good = []

for match1,match2 in matches:
    if match1.distance < 0.75*match2.distance:
        good.append([match1])

flann_matches = cv.drawMatchesKnn(reeses,kp1,cereals,kp2,good,None,flags=2)
display(flann_matches)
