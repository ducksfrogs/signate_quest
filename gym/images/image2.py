from skimage import io
from  skimage import transform

img = io.imread('cat.jpg')

img_trim = img[100:300, 200:400,:]

import numpy as np

tmp = np.array([1,2,3,4])

print(tmp[::-1])

img_resize = transform.resize(img, output_shape=(100, 100))
img_rotate = transform.rotate(img, angle=90)

rot = transform.AffineTransform(translation=(40, 40))

img_affine = transform.warp(img, rot)
