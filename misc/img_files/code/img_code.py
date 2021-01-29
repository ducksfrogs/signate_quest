from skimage import io
from skimage import transform
import matplotlib.pyplot as plt

img = io.imread('')

plt.subplot(1,3,1)
plt.imshow(img)
plt.subplot(1,3,2)
plt.imshow(img)
plt.subplot(1,3,3)
plt.imshow(img_trim)

img_trim = img[100:300, 200:400,:]
plt.imshow(img_trim)

tmp = np.array([1,2,3,4,5])
tmp[::-1]

img_resize = transform.resize(img, output_shape=(100,100))
