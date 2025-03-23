import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

"""
##### HSV -- HLS #####
img = cv.imread('../DATA/00-puppy.jpg')

img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img)
# img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
img = cv.cvtColor(img, cv.COLOR_RGB2HLS)
print(img.shape)
plt.imshow(img)
plt.show()"""

"""###   Image blending   ###
img1 = cv.imread('../DATA/dog_backpack.png')
img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
img2 = cv.imread('../DATA/watermark_no_copy.png')
img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)

# Blending images of the same size
# Images initially have different size let's fix it
img1 = cv.resize(img1,(1200,1200))
img2 = cv.resize(img2,(1200,1200))

blended = cv.addWeighted(src1=img1,alpha=0.8,
                         src2=img2,beta=0.1,gamma=0)
plt.imshow(blended)
plt.show()

# OVERLAY A SMALL IMAGE ON TOP OF A LARGER IMAGE (NO BLENDING)
# Numpy reassignment
img1 = cv.imread('../DATA/dog_backpack.png')
img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
img2 = cv.imread('../DATA/watermark_no_copy.png')
img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)

img2 = cv.resize(img2, (600,600))

large_img, small_img = img1, img2
x_offset = 0
y_offset = 0
x_end = x_offset + small_img.shape[1]
y_end = y_offset + small_img.shape[0]

# BLEND TOGETHER IMAGES OF DIFF SIZES
large_img[y_offset:y_end,x_offset:x_end] = small_img
plt.imshow(large_img)
plt.show()"""

"""## BEAUTIFUL BLENDING
img1 = cv.imread('../DATA/dog_backpack.png')
img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
img2 = cv.imread('../DATA/watermark_no_copy.png')
img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)

img2 = cv.resize(img2, (600,600))

# find the region of interest (roi) in large image:
# print(img1.shape)
x_offset = img1.shape[1] - 600
y_offset = img1.shape[0] - 600

rows,cols,channels = img2.shape

roi = img1[y_offset:1401,x_offset:943]

# create the mask
img2gray = cv.cvtColor(img2,cv.COLOR_RGB2GRAY)

mask_inv = cv.bitwise_not(img2gray) # reverse black to white
# mask_inv no longer contains image channels (check shape)

white_background = np.full(img2.shape,255,dtype=np.uint8)

bk = cv.bitwise_or(src1=white_background,src2=white_background,mask=mask_inv)

# PLACE ORIGINAL IMAGE ON TOP OF THE MASK
# fg = foreground
fg = cv.bitwise_or(img2, img2, mask=mask_inv)

final_roi = cv.bitwise_or(src1=roi,src2=fg)
plt.imshow(final_roi)
plt.show()

# NOW BLEND THE MASKED WITH THE ORIGINAL
large_img, small_img = img1, final_roi
large_img[y_offset:y_offset+small_img.shape[0],x_offset:x_offset+small_img.shape[1]] = small_img
plt.imshow(large_img)
plt.show()

# FOR FURTHER KNOWLEDGE ON BLUNDER AND MANUAL PHOTOSHOPPING THERE ARE LINKES IN THE FOLDER
"""

"""### IMAGE THRESHOLDING ###
# We use image thresholding to minimize space
#  reserved by image {make image binary}
img = cv.imread('../DATA/rainbow.jpg', 0)
plt.imshow(img, cmap='gray')
plt.show()

# below thresh total black
# thresh1 -> image
ret, thresh1 = cv.threshold(img, thresh=127, maxval=255, type=cv.THRESH_BINARY)
# # with inv turns black to white and opposite
# ret,thresh1 = cv.threshold(img,thresh=127,maxval=255,type=cv.THRESH_BINARY_INV)
# # if src is above threshold take it down to thresh
# ret,thresh1 = cv.threshold(img,thresh=127,maxval=255,type=cv.THRESH_TRUNC)
ret, thresh1 = cv.threshold(img, thresh=127, maxval=255, type=cv.THRESH_TOZERO)

plt.imshow(thresh1, cmap='gray')
plt.show()

img = cv.imread('../DATA/crossword.jpg', 0)
plt.imshow(img, cmap='gray')
plt.show()


# i other ide this func is useful to disp image in larger figure
def show_pic(img):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')


# show_pic(img)
ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
plt.imshow(img, cmap='gray')
plt.show()

# Adaptive threshold
th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,11,8)
plt.imshow(th2, cmap='gray')
plt.show()"""

"""def load_img():
    img = cv.imread('../DATA/bricks.jpg').astype(np.float32) / 255
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


def plot(img):
    plt.imshow(img)
    plt.show()


# BRIGHTNESS
img = load_img()
## GAMMA CHANGES BRIGHTNESS
# If gamma << than 1 brighter
# If gamma >> than 1 darker
gamma = 1 / 4
result = np.power(img, gamma)
# plot(result)

img = load_img()
font = cv.FONT_HERSHEY_COMPLEX
cv.putText(img, text='bricks', org=(10, 600), fontFace=font, fontScale=10, color=(255, 0, 0), thickness=4)
plot(img)
kernel = np.ones(shape=(5,5),dtype=(np.float32)) / 25
destination = cv.filter2D(img,-1, kernel)
plot(destination)

img = load_img()
font = cv.FONT_HERSHEY_COMPLEX
cv.putText(img, text='bricks', org=(10, 600), fontFace=font, fontScale=10, color=(255, 0, 0), thickness=4)
# default blur kernel
blurred = cv.blur(img,ksize=(5,5))
plot(blurred)

img = load_img()
font = cv.FONT_HERSHEY_COMPLEX
cv.putText(img, text='bricks', org=(10, 600), fontFace=font, fontScale=10, color=(255, 0, 0), thickness=4)
# gaussian blur
gaus_blur = cv.GaussianBlur(img,(5,5),10)
plot(gaus_blur)

img = load_img()
font = cv.FONT_HERSHEY_COMPLEX
cv.putText(img, text='bricks', org=(10, 600), fontFace=font, fontScale=10, color=(255, 0, 0), thickness=4)
# median blur
median_blur = cv.medianBlur(img,5)
plot(median_blur)

img = load_img()
font = cv.FONT_HERSHEY_COMPLEX
cv.putText(img, text='bricks', org=(10, 600), fontFace=font, fontScale=10, color=(255, 0, 0), thickness=4)
# bilateral filter
bi_fil = cv.bilateralFilter(img,9,75,75)
plot(bi_fil)

############### APPLICATIONS

img = cv.imread('../DATA/sammy.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

noise_img = cv.imread('../DATA/sammy_noise.jpg')
plot(noise_img)

# to remove noise use median blur
median = cv.medianBlur(noise_img,5)
plot(median)"""


## morphological operators { currenlty with errors }
def load_img():
    blank_img = int(np.zeros((600, 600)))
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(blank_img, text='ABCDE', org=(50, 300), fontFace=font, fontScale=5, color=(255, 255, 255), thickness=4)


def plot(img):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap='gray')


img = load_img()
plot(img)
