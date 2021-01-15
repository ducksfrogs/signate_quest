import  matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from PIL import ImageFilter

im = Image.open('input`/')
new_image = im.point(lambda x: x*1.5)
plt.imshow(new_image)

new_image = im.resize((60,60)).resize((300, 300))
new_image2 = im.filter(ImageFilter.GaussianBlur(6))

im = im.filter(ImageFilter.FIND_EDGES)
