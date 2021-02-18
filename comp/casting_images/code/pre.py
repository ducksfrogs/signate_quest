from PIL import Image

im = Image.open('../in)

im1 = Image.open('../input/train_data/')
im2 = Image.open('../input/train_data/')
im3 = Image.open('../input/train_data/')
im4 = Image.open('../input/train_data/')

plt.subplot(1,4,1)
plt.imshow(im1)
plt.title("OK1")

plt.subplot(1,4,2)
plt.imshow(im2)
plt.title("OK2")

plt.subplot(1,4,3)
plt.imshow(im3)
plt.title("NG1")

plt.subplot(1,4,4)
plt.imshow(im4)
plt.title("NG2")

plt.tight_layout()

from PIL import Image, ImageDraw

d = ImageDraw.Draw(im)
d.rectangle(xy=[(40,80),(110,140)], outline='red', width=3)

im4 = ImageDraw.Draw(im4)
d.ellipse(xy=[(10,10),(200,200)], outline='red', width=3)

new_img = im.transpose(Image.FLIP_LEFT_RIGHT)
plt.imshow(new_img)

new_img2 = im.transpose(Image.FLIP_TOP_BOTTOM)
plt.imshow(new_img2)

from PIL import ImageFilter

im = Image.open("../input/train_data")

temp_im = im.point(lambda x: x*1.9)
new_img = temp_im.filter(ImageFilter.GaussianBlur(9))

plt.subplot(1,3,1)
plt.imshow(im)

plt.subplot(1,3,2)
plt.imshow(new_img)

plt.subplot(1,3,3)
plt.imshow(temp_im)

plt.tight_layout()

im = Image.open('../input/train_data/')

new_img = im.resize((60,60)).resize((300,300))
plt.imshow(new_img)

new_im = im.filter(ImageFilter.FIND_EDGES)
plt.imshow(new_im)
