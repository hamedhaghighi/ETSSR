import os
from turtle import down
import numpy as np
import cv2

root_dir ='/home/haghig_h@WMGDS.WMG.WARWICK.AC.UK/Phd_datasets/iPASSR/data/testx2/AirSim'
dest_dir ='/home/haghig_h@WMGDS.WMG.WARWICK.AC.UK/Phd_datasets/iPASSR/data/train/AirSim'
#root_dir = '/media/oem/Local Disk/Phd-datasets/iPASSR/data/testx2/AirSim'
#dest_dir = '/media/oem/Local Disk/Phd-datasets/iPASSR/data/train/AirSim'

def modcrop(img, scale):
    h , w = img.shape[0], img.shape[1]
    return img[ : int((h//scale)*scale), :int((w//scale) * scale)]


def downsample(img):
    img_d = cv2.resize(img.astype('uint8'), (int(0.5 * img.shape[1]), int(0.5 * img.shape[0])), interpolation=cv2.INTER_CUBIC)
    img_d = img_d.astype('float32')
    img_d[:, :, 3] = cv2.resize(img[:, :, 3], (int(0.5 * img.shape[1]), int(0.5 * img.shape[0])))
    return img_d

# Create dataset folder
scale = 2
idx_patch = 1
h_patch, w_patch = 30 , 90
stride = 40
# Get list of scenes in Milddlebury's stereo training dataset and iterate through them
for env_folder in os.listdir(root_dir):
    env_path = os.path.join(root_dir, env_folder)
    img_folders = sorted(os.listdir(env_path))
    for image_folder in img_folders[:-len(img_folders)//10]:
        
        image_path = os.path.join(env_path, image_folder)
        img_hr_0 = np.load(image_path + '/hr0.npy')
        img_hr_1 = np.load(image_path + '/hr1.npy')
        # # Scene data class contains the following data:
        # img_hr_0 = modcrop(left_image, scale)
        # img_hr_1 = modcrop(right_image, scale)
        img_lr_0 = np.load(image_path + '/lr0.npy') 
        img_lr_1 = np.load(image_path + '/lr1.npy')

        for x_lr in range(2, img_lr_0.shape[0] - (h_patch + 2), stride):
            for y_lr in range(2, img_lr_0.shape[1] - (w_patch + 2), stride):
                x_hr = x_lr * scale
                y_hr = y_lr * scale
                hr_patch_0 = img_hr_0[x_hr: (x_lr + h_patch)*scale, y_hr: (y_lr + w_patch)*scale]
                hr_patch_1 = img_hr_1[x_hr: (x_lr + h_patch)*scale, y_hr: (y_lr + w_patch)*scale]
                lr_patch_0 = img_lr_0[x_lr: x_lr + h_patch, y_lr: y_lr + w_patch]
                lr_patch_1 = img_lr_1[x_lr: x_lr + h_patch, y_lr: y_lr + w_patch]
                dst_img_folder = os.path.join(dest_dir, 'patches_x{:d}/{:06d}'.format(scale, idx_patch)) 
                os.makedirs(dst_img_folder, exist_ok=True)
                np.save(dst_img_folder + '/hr0.npy', hr_patch_0)
                np.save(dst_img_folder + '/hr1.npy', hr_patch_1)
                np.save(dst_img_folder + '/lr0.npy', lr_patch_0)
                np.save(dst_img_folder + '/lr1.npy', lr_patch_1)
                print('writing patch id:', idx_patch)
                idx_patch = idx_patch + 1
    
    # ground_truth_disp_image = scene_data.disp_image
    
    # ndisp = scene_data.ndisp

    # # Start timer
    # timer = Timer()
    # timer.start()
    # # Simluate match result by adding a bit of noise to the ground truth
    # # REPLACE THIS WITH THE RESULT FROM YOUR STEREO ALGORITHM
    # # e.g. test_disp_image = cv2.imread("disp_result.tif",cv2.IMREAD_UNCHANGED)
    # noise = np.random.uniform(low=0, high=3.0, size=ground_truth_disp_image.shape)
    # test_disp_image = ground_truth_disp_image + noise
    # # Record elapsed time for simulated match
    # elapsed_time = timer.elapsed()

    # # Store match results
    # match_result = MatchData.MatchResult(
    #     left_image,right_image,ground_truth_disp_image,test_disp_image,elapsed_time,ndisp)
    # # Create match data (which includes scene data needed for rank comparision in eval)
    # match_data = MatchData(scene_info,match_result)
    # # Add match data to list
    # match_data_list.append(match_data)
