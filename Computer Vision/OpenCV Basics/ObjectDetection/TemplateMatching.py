import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# TEMPLATE MATCHING
# check if target image is in larger image
full = cv.imread('../DATA/sammy.jpg')
full = cv.cvtColor(full, cv.COLOR_BGR2RGB)
plt.imshow(full)
plt.show()

face = cv.imread('../DATA/sammy_face.jpg')
face = cv.cvtColor(face, cv.COLOR_BGR2RGB)

# you should have already the exact image you are
# looking for in the larger image(exactly the siz

methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF',
           'cv.TM_SQDIFF_NORMED']

for m in methods:
    # CREATE A COPY OF IMGAGE
    full_copy = full.copy()

    method = eval(m)

    # TEMPLATE MATCHING
    # res is a heatmap with high value where face is
    res = cv.matchTemplate(full_copy, face, method)

    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    if method in [cv.TM_SQDIFF,cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    height, width, channels = face.shape

    bottom_right = (top_left[0]+width, top_left[1]+height)

    cv.rectangle(full_copy, top_left, bottom_right,(0,0,255),10)

    plt.subplot(121)
    plt.imshow(res)
    plt.title('HEATMAP OF TEMPLATE MATCHING')
    plt.subplot(122)
    plt.imshow(full_copy)
    plt.title('DETECTION OF TEMPLATE')
    plt.suptitle(m) # title everything with the method used
    plt.show()



"""# eval function:
# evaluate string as function: example:
mystring = 'sum'
print(eval(mystring))
myfunc = eval(mystring)
print(myfunc([1,2,3]))"""
