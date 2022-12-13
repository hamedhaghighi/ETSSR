import os

import cv2
import numpy as np


def modcrop(img, scale):
    h, w = img.shape[0], img.shape[1]
    return img[: int((h // scale) * scale), :int((w // scale) * scale)]


def load_sensor_data(filename, c_name):
    if c_name == 'disp_':
        img = np.load(filename)['a'][..., None]
    else:
        img = cv2.imread(filename).astype('float32')
    return img


def downsample(img):
    img_d = cv2.resize(img.astype('uint8'),
                       (int(0.5 * img.shape[1]),
                        int(0.5 * img.shape[0])),
                       interpolation=cv2.INTER_CUBIC)
    img_d = img_d.astype('float32')
    img_d[:, :, 3] = cv2.resize(
        img[:, :, 3], (int(0.5 * img.shape[1]), int(0.5 * img.shape[0])))
    return img_d


if __name__ == '__main__':
    root_dir = '/home/haghig_h@WMGDS.WMG.WARWICK.AC.UK/Phd_datasets/iPASSR/data/Carla'
    dest_dir = '/home/haghig_h@WMGDS.WMG.WARWICK.AC.UK/Phd_datasets/iPASSR/data/Carla_patches'
    scale = 2
    h_patch, w_patch = 90, 160
    stride_h, stride_w = 90, 160
    cam_names =  ['','disp_', 'seg_']
    exts = ['.png','.npz','.png']

    for n_cam in [0, 1]:
        idx_patch = 1
        for env_folder in os.listdir(root_dir):
            env_path = os.path.join(root_dir, env_folder)
            img_folders = sorted(os.listdir(env_path))
            for image_folder in img_folders[:-len(img_folders) // 10]:
                image_path = os.path.join(env_path, image_folder)
                img_x1_list = []
                img_x2_list = []
                img_x4_list = []
                for c_name, ext in zip(cam_names, exts):
                    def load_mono_cam(scale):
                        filename = os.path.join(
                            image_path, 'imgx{}_{}{}{}'.format(
                                scale, c_name, n_cam, ext))
                        img = load_sensor_data(filename, c_name)
                        return img
                    img_x1_list.append(load_mono_cam(1))
                    img_x2_list.append(load_mono_cam(2))
                    img_x4_list.append(load_mono_cam(4))

                img_x1 = np.concatenate(img_x1_list, axis=-1)
                img_x2 = np.concatenate(img_x2_list, axis=-1)
                img_x4 = np.concatenate(img_x4_list, axis=-1)

                for x_x4 in range(2, img_x4.shape[0] - (h_patch + 2), stride_h):
                    for y_x4 in range(
                            2, img_x4.shape[1] - (w_patch + 2), stride_w):
                        x_x2, y_x2 = x_x4 * 2, y_x4 * 2
                        x_x1, y_x1 = x_x4 * 4, y_x4 * 4
                        x4_patch = img_x4[x_x4: x_x4 +
                                        h_patch, y_x4: y_x4 + w_patch]
                        x2_patch = img_x2[x_x2: (
                            x_x4 + h_patch) * 2, y_x2: (y_x4 + w_patch) * 2]
                        x1_patch = img_x1[x_x1: (
                            x_x4 + h_patch) * 4, y_x1: (y_x4 + w_patch) * 4]

                        dst_img_folder = os.path.join(
                            dest_dir, 'patches/{:06d}'.format(idx_patch))
                        os.makedirs(dst_img_folder, exist_ok=True)

                        def save_mono_cam(img, scale):
                            filename = os.path.join(
                                dst_img_folder, 'imgx{}_{}.png'.format(
                                    scale, n_cam))
                            cv2.imwrite(filename, img[..., :3].astype('uint8'))
                            filename = os.path.join(
                                dst_img_folder, 'imgx{}_disp_{}'.format(
                                    scale, n_cam))
                            np.savez_compressed(filename, a=img[..., 3])
                            filename = os.path.join(
                                dst_img_folder, 'imgx{}_seg_{}.png'.format(
                                    scale, n_cam))
                            cv2.imwrite(filename, img[..., 4:7].astype('uint8'))

                        save_mono_cam(x1_patch, 1)
                        save_mono_cam(x2_patch, 2)
                        save_mono_cam(x4_patch, 4)

                        print('cam no:', n_cam, 'writing patch id:', idx_patch)
                        idx_patch = idx_patch + 1
