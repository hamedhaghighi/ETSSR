import glob
from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt

# left, top , right , bottom = 1026, 513, 1612, 1014
left, top, right, bottom = 936, 513, 1522, 1014
dir = '../checkpoints/Town02_img_9_right/'
# img = Image.open(os.path.join(dir, 'hr_right_Town02_img_9.png'))
# draw = ImageDraw.Draw(img)
# draw.rectangle(
#     (left, top, right, bottom),
#     fill=None,
#     outline=(255, 0, 0), width=3)
# img.show()
img_files = glob.glob(dir + '*.png')
for img_file in os.listdir(dir):
    # img = plt.imread(os.path.join(dir, img_file))
    # plt.imshow(img)
    # plt.show()
    img = Image.open(os.path.join(dir, img_file))
    img = img.crop((left, top, right, bottom))
    img.save(os.path.join(dir, 'Cropped_' + img_file))
    

