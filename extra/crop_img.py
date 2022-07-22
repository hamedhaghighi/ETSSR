from cmath import exp
import glob
from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt
import shutil
# left, top , right , bottom = 1026, 513, 1612, 1014


def show_img_for_crop():
    img = Image.open(os.path.join(dir, 'hr_right_Town02_img_9.png'))
    draw = ImageDraw.Draw(img)
    draw.rectangle(
        (left, top, right, bottom),
        fill=None,
        outline=(255, 0, 0), width=3)
    img.show()

def crop_and_save_images(dir, search_name, left, top, right, bottom):

    for img_file_name in os.listdir(dir):
        if search_name in img_file_name and not 'Cropped_' in img_file_name:
            img = Image.open(os.path.join(dir, img_file_name))
            img = img.crop((left, top, right, bottom))
            img.save(os.path.join(dir, 'Cropped_' + img_file_name))

    return


def filter_and_copy(src_dir, dst_dir, search_name, exp_name):

    for filename in os.listdir(src_dir):
        dst_filename = filename.split('.')[0] + '_' + exp_name + '.png'
        if search_name in filename:
            shutil.copy(os.path.join(src_dir, filename), os.path.join(dst_dir, dst_filename))
    return


def copy_from_checkpoints_to_all_results(filename, dir, exp_names):
    
    dst_dir = os.path.join(dir, 'all_results/' +  filename)
    os.makedirs( dst_dir, exist_ok=True) 
    for exp_name in exp_names:
        exp_result_dir = os.path.join(dir, exp_name + '/results/')
        filter_and_copy(exp_result_dir, dst_dir, filename, exp_name)

    return

def crop_imgs_in_a_folder(dir, filename):
    
    all_results_img_dir = os.path.join(dir, filename)
    crop_and_save_images(all_results_img_dir, filename, left, top, right, bottom)


def plot_crop_images(dir, n):

    i = 0
    for img_filename in os.listdir(dir):
        if 'Cropped_' in img_filename:
            plt.subplot(1, n , i + 1)
            title = img_filename.split('.')[0].split('_')[-1]
            plt.title(title)
            img_array = plt.imread(os.path.join(dir, img_filename))
            plt.imshow(img_array)
            i = i + 1

    plt.show()

if __name__ == '__main__':
    filename = 'left_Town01_img_92'
    left, top, right, bottom = 936, 513, 1522, 1014
    exp_names = ['bicubic', 'Ipassr', 'PASSRnet', 'RCAN', 'VDSR', 'transformer_all_swinx4_CP_L4']
    dir = '../checkpoints/'
    copy_from_checkpoints_to_all_results(filename, dir, exp_names)
    dir = '../checkpoints/all_results/'
    crop_imgs_in_a_folder(dir, filename)
    all_results_img_dir = os.path.join(dir, filename)
    plot_crop_images(all_results_img_dir, 7)

