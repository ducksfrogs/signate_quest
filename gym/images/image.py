from skimage import io

from skimage import transform

img = io.imread("cat.jpg")

from matplotlib import pyplot as plt

plt.imshow(img)


plt.subplot(1,3,1)
plt.imshow(img)

plt.subplot(1,3,2)
plt.imshow(img)

plt.subplot(1,3,3)
plt.imshow(img)
