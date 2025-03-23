import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

flat_chess = cv.imread('../DATA/flat_chessboard.png')

found,corners = cv.findChessboardCorners(flat_chess,(7,7))
# found is boolean to indicate if found
cv.drawChessboardCorners(flat_chess,(7,7),corners,found)

plt.imshow(flat_chess)
plt.show()

dots = cv.imread('../DATA/dot_grid.png')
found,corners = cv.findCirclesGrid(dots,(10,10),cv.CALIB_CB_SYMMETRIC_GRID)

cv.drawChessboardCorners(dots,(10,10),corners,found)

plt.imshow(dots)
plt.show()
