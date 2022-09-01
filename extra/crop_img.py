from cmath import exp
import glob
from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt
import shutil
# left, top , right , bottom = 1026, 513, 1612, 1014
import math
import cv2 
import matplotlib.gridspec as gridspec

def show_img_for_crop(dir, filename):
     # Read image
    for img_filename in os.listdir(dir):
        if filename +'_' in img_filename:
            filename = img_filename
            break

    im = cv2.imread(os.path.join(dir, filename))
    # plt.imshow(im)
    # plt.show()
    # Select ROI
    r = cv2.selectROI(im)
    # if 'right' in filename:
    #     r = [2049, 727, 2049 + 2213 - 2081, 727 + 817 - 738]
    # else :
    #     r = [2081, 738, 2213 - 2081, 817 - 738]
    # Crop image
    imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    
    # Display cropped image
    cv2.imshow("Image", imCrop)
    cv2.waitKey(0)
    return r[0] , r[1] , r[0] + r[2], r[1] + r[3]

def crop_and_save_images(dir, search_name, left, top, right, bottom , view):

    for img_file_name in os.listdir(dir):
        if  not 'Cropped_' in img_file_name and view in img_file_name:
            img = Image.open(os.path.join(dir, img_file_name))
            img = img.crop((left, top, right, bottom))
            img.save(os.path.join(dir, 'Cropped_' + img_file_name))

    return




def copy_from_checkpoints_to_all_results(search_name, dir, exp_names, dst_name=None):
    
    dst_dir = os.path.join(dir, 'all_results/' +  search_name) if dst_name is None else os.path.join(dir, 'all_results/' +  dst_name)
    os.makedirs( dst_dir, exist_ok=True) 
    for exp_name in exp_names:
        if 'Bicubic' in exp_name:
            exp_result_dir = os.path.join(dir, 'Bicubic' + '/results/')
        else:
            exp_result_dir = os.path.join(dir, exp_name + '/results/')

        for filename in os.listdir(exp_result_dir):
            if 'Bicubic' in exp_name:
                search_name_t = exp_name.split('_')[1] + '_' + search_name + '.' if 'Bicubic_hr' in exp_name else exp_name.split('_')[1] + '_' + search_name + '_'
                if search_name_t in filename:
                    if 'Bicubic_hr' in exp_name:
                        dst_filename = filename[:-4] + '_' + 'HR' + '.png'
                    else:
                        dst_filename = filename[:-4] + '_' + 'Bicubic' + '.png'
                    shutil.copy(os.path.join(exp_result_dir, filename), os.path.join(dst_dir, dst_filename))
            else:
                if search_name + '_' in filename:
                    dst_filename = filename[:-4] + '_' + exp_name + '.png'
                    shutil.copy(os.path.join(exp_result_dir, filename), os.path.join(dst_dir, dst_filename))

    return


def split_file_name(filename):

    filename_list = filename.split('_')
    search_name = '_'.join(filename_list[0:5])
    psnr = float(filename_list[5])
    return search_name, psnr

def find_best_result_img(dir, exp_names):
    main_exp_name = 'ETSSR'
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


def crop_imgs_in_a_folder(dir, filename, left, top, right, bottom, view):
    
    all_results_img_dir = os.path.join(dir, filename)
    crop_and_save_images(all_results_img_dir, filename, left, top, right, bottom, view)


def plot_crop_images(dir, n, exp_names, im):
    # plt.figure(0)
    # plt.axis('off')
    # plt.imshow(im)
    fig = plt.figure(1)
    # gs1.update(wspace=0.025, hspace=0.01) # set the spacing between axes. 
    psnr_list = []
    ssim_list = []
    i = 0
    f_size = 14
    axs =  fig.subplots(2, n).flatten()
    for j, view in enumerate(['left', 'right']):
        for exp_name in exp_names:
            for img_filename in os.listdir(dir):
                if exp_name in img_filename and 'Cropped_' in img_filename and view in img_filename:
                    axs[i].spines['top'].set_visible(False)
                    axs[i].spines['right'].set_visible(False)
                    axs[i].spines['bottom'].set_visible(False)
                    axs[i].spines['left'].set_visible(False)
                    axs[i].get_xaxis().set_ticks([])
                    axs[i].get_yaxis().set_ticks([])
                    # ax.set_aspect('equal')
                    psnr = float(img_filename.split('_')[6]) if exp_name!= 'HR' else 0
                    ssim =  float(img_filename.split('_')[7]) if exp_name!= 'HR' else 0

                    if i == 0:
                        axs[i].set_ylabel('Left', fontsize=f_size)
                    if i == n:
                        axs[i].set_ylabel('Right', fontsize=f_size)
                    if view == 'left':
                        psnr_list.append(psnr); ssim_list.append(ssim)
                    else:
                        exp_name = img_filename.split('_')[-1][:-4]
                        title = '{:.2f}'.format((psnr + psnr_list[i - n])/2) + ', ' + '{:.3f}'.format((ssim + ssim_list[i - n]) / 2) if exp_name!= 'HR' else ''
                        # plt.title(exp_name + '\n' + title, y =-0.01)
                        axs[i].set_xlabel(exp_name + '\n' + title, fontsize=f_size)
                    # plt.axis('off')
                    img_array = plt.imread(os.path.join(dir, img_filename))
                    axs[i].imshow(img_array)
                    i = i + 1
    plt.subplots_adjust(wspace = 0.04, hspace= 0.0, bottom = 0.368, top=0.538, left=0.125, right=0.9)
    plt.show()

    
if __name__ == '__main__':
    option = 'plot_crop_images'
    # option = 'plot'

    total_filename = 'Town01_img_29'
    # option = 'plot'
    if 'plot' in option:
        filename = 'left_' + total_filename
        filename_2 = 'right_' + total_filename
        if 'crop_images' in option:
            dir = '../checkpoints/ETSSR/results/'
            left, top, right, bottom = show_img_for_crop(dir, filename)
            left_1, top_1, right_1, bottom_1 = show_img_for_crop(dir, filename_2)

            exp_names = ['Bicubic_sr','Bicubic_hr', 'iPASSR', 'PASSRnet', 'RCAN', 'VDSR', 'ETSSR', 'RDN', 'EDSR', 'SSRDEFNet', 'StereoSR']
            dir = '../checkpoints/'
            copy_from_checkpoints_to_all_results(filename, dir, exp_names)
            copy_from_checkpoints_to_all_results(filename_2, dir, exp_names, filename)
            dir = '../checkpoints/all_results/'
            
            crop_imgs_in_a_folder(dir, filename, left, top, right, bottom, 'left')
            crop_imgs_in_a_folder(dir, filename, left_1, top_1, left_1 + (right - left), top_1 + (bottom - top), 'right')

        dir = '../checkpoints/all_results/'
        all_results_img_dir = os.path.join(dir, filename)
        # exp_names = ['Bicubic', 'VDSR', 'EDSR', 'RCAN', 'RDN', 'StereoSR', 'PASSRnet','iPASSR','SSRDEFNet','ETSSR','HR']
        exp_names = ['Bicubic', 'VDSR','RCAN','EDSR','RDN', 'StereoSR', 'PASSRnet','SSRDEFNet','iPASSR','ETSSR','HR']
        # for fn in os.listdir(all_results_img_dir):
        #     if 'Cropped' not in fn and 'HR' in fn and 'left' in fn:
        #         im = plt.imread(os.path.join(all_results_img_dir, fn))
        #         im = cv2.rectangle(im, (left, top), (right, bottom),  (255 ,0, 0), 2)
        im = None
        plot_crop_images(all_results_img_dir, len(exp_names), exp_names, im)

    elif option == 'find_max':
        exp_names = ['iPASSR', 'PASSRnet', 'RCAN', 'VDSR', 'RDN', 'EDSR', 'SSRDEFNet', 'StereoSR']
        dir = '../checkpoints/'
        find_best_result_img(dir, exp_names)
        

    # names = ['sr_left_Town07_img_1', 'town02_img_9', 'town07_img_96','town07_img_97', 'town01_92', 'town05_28']