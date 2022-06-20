import glob
from PIL import Image
import os


left, top , right , bottom = 100, 100, 300, 300
dir = './checkpoints_4/bicubic/results/'
img_files = glob.glob(dir + '*.png')
for img_file in img_files:
    img = Image.open(img_file)
    img = img.crop(left, top, right, bottom)
    img.save(os.path.join(dir, img_file))
    

