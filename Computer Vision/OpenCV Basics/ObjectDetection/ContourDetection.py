import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('../DATA/internal_external.png', 0)

loveyou, hierarchy = cv.findContours(img, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

external_contours = np.zeros(img.shape)

for i in range(len(loveyou)):

    # external contour
    if hierarchy[0][i][3] == -1:
        cv.drawContours(external_contours, loveyou, i, 255, -1)

internal_contours = np.zeros(img.shape)

for i in range(len(loveyou)):

    # external contour
    if hierarchy[0][i][3] != -1:
        cv.drawContours(internal_contours, loveyou, i, 255, -1)

plt.imshow(external_contours, cmap='gray')
plt.show()

plt.imshow(internal_contours, cmap='gray')
plt.show()

plt.imshow(img, cmap='gray')
plt.show()
