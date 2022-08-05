from cmath import exp
import glob
from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt
import shutil
# left, top , right , bottom = 1026, 513, 1612, 1014
import math
import cv2 


def show_img_for_crop(dir, filename):
     # Read image
    for img_filename in os.listdir(dir):
        if filename in img_filename:
            filename = img_filename
            break
    
    im = cv2.imread(os.path.join(dir, filename))

    # Select ROI
    r = cv2.selectROI(im)

    # Crop image
    imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

    # Display cropped image
    cv2.imshow("Image", imCrop)
    cv2.waitKey(0)
    
    return r[0] , r[1] , r[0] + r[2], r[1] + r[3]

def crop_and_save_images(dir, search_name, left, top, right, bottom):

    for img_file_name in os.listdir(dir):
        if search_name in img_file_name and not 'Cropped_' in img_file_name:
            img = Image.open(os.path.join(dir, img_file_name))
            img = img.crop((left, top, right, bottom))
            img.save(os.path.join(dir, 'Cropped_' + img_file_name))

    return



    
    return


def copy_from_checkpoints_to_all_results(search_name, dir, exp_names):
    
    dst_dir = os.path.join(dir, 'all_results/' +  search_name)
    os.makedirs( dst_dir, exist_ok=True) 
    for exp_name in exp_names:
        if 'bicubic' in exp_name:
            exp_result_dir = os.path.join(dir, 'bicubic' + '/results/')
        else:
            exp_result_dir = os.path.join(dir, exp_name + '/results/')

        for filename in os.listdir(exp_result_dir):
            if 'bicubic' in exp_name:
                search_name_t = exp_name.split('_')[1] + '_' + search_name
                if search_name_t in filename:
                    if 'bicubic_hr' in exp_name:
                        dst_filename = filename[:-4] + '_' + 'HR' + '.png'
                    else:
                        dst_filename = filename[:-4] + '_' + 'bicubic' + '.png'
                    shutil.copy(os.path.join(exp_result_dir, filename), os.path.join(dst_dir, dst_filename))
            else:
                if search_name in filename:
                    dst_filename = filename[:-4] + '_' + exp_name + '.png'
                    shutil.copy(os.path.join(exp_result_dir, filename), os.path.join(dst_dir, dst_filename))

    return


def split_file_name(filename):

    filename_list = filename.split('_')
    search_name = '_'.join(filename_list[0:5])
    psnr = float(filename_list[5])
    return search_name, psnr

def find_best_result_img(dir, exp_names):
    main_exp_name = 'transformer_all_swinx4_CP_L4'
    main_exp_result_dir = os.path.join(dir, main_exp_name + '/results/')
    img_psnr_diff_dict = dict()
    for img_filename in os.listdir(main_exp_result_dir):
        min_diff = 100
        min_exp_name = ''
        search_name , psnr = split_file_name(img_filename)
        for exp_name in exp_names:
            exp_result_dir = os.path.join(dir, exp_name + '/results/')
            for img_filename_2 in os.listdir(exp_result_dir):
                if search_name in img_filename_2:
                    _ , psnr_2 = split_file_name(img_filename_2)
                    if psnr - psnr_2 < min_diff:
                        min_diff = psnr - psnr_2
                        min_exp_name = exp_name
        img_psnr_diff_dict[search_name + '_' + min_exp_name ] = math.floor(min_diff * 1000) / 1000
    img_psnr_diff_dict = {k: v for k, v in sorted(img_psnr_diff_dict.items(), key=lambda item: item[1])}
    print(img_psnr_diff_dict)
    return


def crop_imgs_in_a_folder(dir, filename):
    
    all_results_img_dir = os.path.join(dir, filename)
    crop_and_save_images(all_results_img_dir, filename, left, top, right, bottom)


def plot_crop_images(dir, n):

    i = 0
    for img_filename in os.listdir(dir):
        if 'Cropped_' in img_filename:
            plt.subplot(1, n , i + 1)
            exp_name = img_filename.split('_')[-1][:-4]
            psnr = img_filename.split('_')[6]
            plt.title(exp_name + '_' + psnr)
            plt.axis('off')
            img_array = plt.imread(os.path.join(dir, img_filename))
            plt.imshow(img_array)
            i = i + 1

    plt.subplots_adjust(wspace=0.1, hspace=0)
    plt.show()

if __name__ == '__main__':
    option = 'crop_images'
    if option == 'crop_images':
        dir = '../checkpoints/transformer_all_swinx4_CP_L4/results/'
        filename = 'left_Town07_img_15'
        left, top, right, bottom = show_img_for_crop(dir, filename)
        exp_names = ['bicubic_sr','bicubic_hr', 'Ipassr', 'PASSRnet', 'RCAN', 'VDSR', 'transformer_all_swinx4_CP_L4']
        dir = '../checkpoints/'
        copy_from_checkpoints_to_all_results(filename, dir, exp_names)
        dir = '../checkpoints/all_results/'
        crop_imgs_in_a_folder(dir, filename)
        all_results_img_dir = os.path.join(dir, filename)
        plot_crop_images(all_results_img_dir, 7)

    elif option == 'find_max':
        exp_names = ['Ipassr', 'PASSRnet', 'RCAN', 'VDSR']
        dir = '../checkpoints/'
        find_best_result_img(dir, exp_names)
        

