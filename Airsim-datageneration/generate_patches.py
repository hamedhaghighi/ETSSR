import os
import numpy as np
import cv2



def modcrop(img, scale):
    h , w = img.shape[0], img.shape[1]
    return img[ : int((h//scale)*scale), :int((w//scale) * scale)]


def load_sensor_data(filename, c_name):
    if c_name == 'disp_':
        img = np.load(filename)['a'][... , None]
    else:
        img = cv2.imread(filename).astype('float32')
    return img

def downsample(img):
    img_d = cv2.resize(img.astype('uint8'), (int(0.5 * img.shape[1]), int(0.5 * img.shape[0])), interpolation=cv2.INTER_CUBIC)
    img_d = img_d.astype('float32')
    img_d[:, :, 3] = cv2.resize(img[:, :, 3], (int(0.5 * img.shape[1]), int(0.5 * img.shape[0])))
    return img_d

# Create dataset folder
dataset = 'Carla'
root_dir = '/home/haghig_h@WMGDS.WMG.WARWICK.AC.UK/Phd_datasets/iPASSR/data/{}'.format(dataset)
dest_dir = '/home/haghig_h@WMGDS.WMG.WARWICK.AC.UK/Phd_datasets/iPASSR/data/{}_patches_L'.format(dataset)
#root_dir = '/media/oem/Local Disk/Phd-datasets/iPASSR/data/testx2/AirSim'
#dest_dir = '/media/oem/Local Disk/Phd-datasets/iPASSR/data/train/AirSim'
scale = 2
h_patch, w_patch = 90 , 160
stride = 40
cam_names = ['', 'disp_', 'seg_', 'sn_'] if dataset == 'AirSim' else ['', 'disp_', 'seg_']
exts = ['.png', '.npz', '.png', '.png'] if dataset == 'AirSim' else ['.png', '.npz', '.png']
# Get list of scenes in Milddlebury's stereo training dataset and iterate through them
for n_cam in [0, 1]:
    idx_patch = 1
    for env_folder in os.listdir(root_dir):
        env_path = os.path.join(root_dir, env_folder)
        img_folders = sorted(os.listdir(env_path))
        for image_folder in img_folders[:-len(img_folders)//10]:
            image_path = os.path.join(env_path, image_folder)
            img_x1_list = []
            img_x2_list = []
            img_x4_list = []
            for c_name, ext in zip(cam_names, exts): 
                def load_mono_cam(scale):
                    filename = os.path.join(image_path, 'imgx{}_{}{}{}'.format(scale, c_name, n_cam, ext))
                    img = load_sensor_data(filename, c_name)
                    return img
                img_x1_list.append(load_mono_cam(1))
                img_x2_list.append(load_mono_cam(2))
                img_x4_list.append(load_mono_cam(4))

            img_x1 = np.concatenate(img_x1_list, axis=-1)
            img_x2 = np.concatenate(img_x2_list, axis=-1)
            img_x4 = np.concatenate(img_x4_list, axis=-1)

            for x_x4 in range(2, img_x4.shape[0] - (h_patch + 2), stride):
                for y_x4 in range(2, img_x4.shape[1] - (w_patch + 2), stride):
                    x_x2, y_x2 = x_x4 * 2, y_x4 * 2
                    x_x1, y_x1 = x_x4 * 4, y_x4 * 4
                    x4_patch = img_x4[x_x4: x_x4 + h_patch, y_x4: y_x4 + w_patch]
                    x2_patch = img_x2[x_x2: (x_x4 + h_patch) * 2, y_x2: (y_x4 + w_patch) * 2]
                    x1_patch = img_x1[x_x1: (x_x4 + h_patch) * 4, y_x1: (y_x4 + w_patch) * 4]

                    dst_img_folder = os.path.join(dest_dir, 'patches/{:06d}'.format(idx_patch)) 
                    os.makedirs(dst_img_folder, exist_ok=True)

                    def save_mono_cam(img, scale):
                        filename = os.path.join(dst_img_folder, 'imgx{}_{}.png'.format(scale, n_cam))
                        cv2.imwrite(filename, img[..., :3].astype('uint8'))
                        filename = os.path.join(dst_img_folder, 'imgx{}_disp_{}'.format(scale, n_cam))
                        np.savez_compressed(filename, a=img[..., 3])
                        filename = os.path.join(dst_img_folder, 'imgx{}_seg_{}.png'.format(scale, n_cam))
                        cv2.imwrite(filename, img[..., 4:7].astype('uint8'))
                        if dataset == 'AirSim':
                            filename = os.path.join(dst_img_folder, 'imgx{}_sn_{}.png'.format(scale, n_cam))
                            cv2.imwrite(filename, img[..., 7:].astype('uint8'))
                    save_mono_cam(x1_patch, 1)
                    save_mono_cam(x2_patch, 2)
                    save_mono_cam(x4_patch, 4)

                    print('cam no:', n_cam, 'writing patch id:', idx_patch)
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
