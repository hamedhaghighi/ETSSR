import os
import numpy as np
import cv2
from stereomideval.structures import MatchData
from stereomideval.dataset import Dataset
from stereomideval.eval import Eval, Timer

root_dir ='/media/oem/Local Disk/Phd-datasets/iPASSR/datasets_Airsim'

def modcrop(img, scale):
    h , w = img.shape[0], img.shape[1]
    return img[ : int((h//scale)*scale), :int((w//scale) * scale)]
    
# Create dataset folder
scale = 2
idx_patch = 1
match_data_list = []
# Get list of scenes in Milddlebury's stereo training dataset and iterate through them
image_folders = os.listdir(root_dir)
for folder in image_folders:
    
    image_path = os.path.join(root_dir, folder)
    left_image = np.load(image_path + '/im0.npy')
    right_image = np.load(image_path + '/im1.npy')
    # # Scene data class contains the following data:
    img_hr_0 = modcrop(left_image, scale)
    img_hr_1 = modcrop(right_image, scale)
    img_lr_0 = cv2.resize(img_hr_0.astype('uint8'), (int(0.5 * img_hr_0.shape[1]) , int(0.5 * img_hr_0.shape[0])), interpolation=cv2.INTER_CUBIC)
    img_lr_1 = cv2.resize(img_hr_1.astype('uint8'), (int(0.5 * img_hr_1.shape[1]), int(0.5 * img_hr_1.shape[0])), interpolation=cv2.INTER_CUBIC)
    img_lr_0 = img_lr_0.astype('float32')
    img_lr_1 = img_lr_1.astype('float32')
    img_lr_0[:,:,3] = cv2.resize(img_hr_0[:,:,3], (int(0.5 * img_hr_0.shape[1]), int(0.5 * img_hr_0.shape[0])))
    img_lr_1[:,:,3] = cv2.resize(img_hr_1[:,:,3], (int(0.5 * img_hr_1.shape[1]), int(0.5 * img_hr_1.shape[0])))
    for x_lr in range(2, img_lr_0.shape[0] - 32, 20):
        for y_lr in range(2, img_lr_0.shape[1] - 92, 20):
            x_hr = x_lr * scale
            y_hr = y_lr * scale
            hr_patch_0 = img_hr_0[x_hr: (x_lr + 30)*scale, y_hr: (y_lr + 90)*scale]
            hr_patch_1 = img_hr_1[x_hr: (x_lr + 30)*scale, y_hr: (y_lr + 90)*scale]
            lr_patch_0 = img_lr_0[x_lr: x_lr + 30, y_lr: y_lr + 90]
            lr_patch_1 = img_lr_1[x_lr: x_lr + 30, y_lr: y_lr + 90]
            dst_root = os.path.join(root_dir, 'patches_x{:d}/{:06d}'.format(scale, idx_patch)) 
            os.makedirs(dst_root, exist_ok=True)
            np.save(dst_root + '/hr0.npy', hr_patch_0)
            np.save(dst_root + '/hr1.npy', hr_patch_1)
            np.save(dst_root + '/lr0.npy', lr_patch_0)
            np.save(dst_root + '/lr1.npy', lr_patch_1)
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
