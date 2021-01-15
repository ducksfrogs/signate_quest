from PIL import  Image

im = Image.open('../input/')

print(im.format)
print(im.size)

import matplotlib.pyplot as plt

plt.imshow(im)

plt.subplot(1,4,1)
plt.imshow(im1)

plt.subplot(1,4,2)
plt.imshow(im2)

plt.subplot(1,4,3)
plt.imshow(im3)

plt.subplot(1,4,4)
pl.imshow(im4)


im = Image.new('RGB', (300,300), 'gray')
d = ImageDraw.Draw(im)
d.rectangle(xy=[(100, 220),(100, 270)], outline='red', width=3)

plt.imshow(im)

d = ImageDraw.Draw(im)
d.rectangle(xy=[(40,80),(110,140)], outline='red', width=3)
plt.imshow(im)

im = Image.new('RGB', (300,300), 'gray')
d = ImageDraw.Draw(im)
d.ellipse(xy=[(10,10),(280,280)], outline='red', width=3)

new_img = im.transpose(Image.FLIP_LEFT_RIGHT)

plt.imshow(new_img)
plt.show()
