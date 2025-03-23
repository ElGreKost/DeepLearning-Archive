import numpy as np
import matplotlib.pyplot as plt  # %matplotlib inline

from PIL import Image

pic = Image.open('../DATA/00-puppy.jpg')

pic_arr = np.asarray(pic)
# pic_arr.shape # 3d dimension are the colors RGB

# plt.imshow(pic_arr)

pic_red = pic_arr.copy()

# RED CHANNEL VALUES 0 no red, pure black -255 full pure red
plt.imshow(pic_red[:, :, 0], cmap='gray')
plt.show()

