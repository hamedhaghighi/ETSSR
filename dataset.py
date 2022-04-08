from PIL import Image
import os
from torch.utils.data.dataset import Dataset
import random
import torch
import numpy as np
import cv2

class DataSetLoader(Dataset):
    def __init__(self, cfg, to_tensor=True):
        super(DataSetLoader, self).__init__()
        self.cfg = cfg
        self.dataset_dir = cfg.data_dir
        self.file_list = sorted(os.listdir(self.dataset_dir))
        self.to_tensor = to_tensor
        self.scale = cfg.scale
        self.c_names = ['', 'disp_', 'seg_']
        self.exts = ['.png', '.npz', '.png']

    def read_img(self, img_path):
        ext = img_path.split('.')[-1]
        if ext == 'npy':
            return np.load(img_path)[..., :self.cfg.input_channel]
        elif ext == 'png':
            assert self.cfg.input_channel == 3
            img = Image.open(img_path)
            return np.array(img,  dtype=np.float32)

    def load_sensor_data(self, filename, c_name):
        if c_name == 'disp_':
            img = np.load(filename)['a']
        else:
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('float32')
        return img


    def __getitem__(self, index):
        image_path = os.path.join(self.dataset_dir, self.file_list[index])
        def load_mono_cam(n_cam, scale):
            img_list = []
            for c_name, ext in zip(self.cam_names, self.exts):
                def load_scale_cam(scale):
                    filename = os.path.join(image_path, 'imgx{}_{}{}{}'.format(scale, c_name, n_cam, ext))
                    img = self.load_sensor_data(filename, c_name)
                    return img
                img_list.append(load_scale_cam(scale))
            return img_list
        if self.scale == 2:
            img_hr_left = np.concatenate(load_mono_cam(0, 2), axis=-1)
            img_hr_right = np.concatenate(load_mono_cam(1, 2), axis=-1)
            img_lr_left = np.concatenate(load_mono_cam(0, 4), axis=-1)
            img_lr_right = np.concatenate(load_mono_cam(1, 4), axis=-1)
        elif self.scale == 4:
            img_hr_left = np.concatenate(load_mono_cam(0, 1), axis=-1)
            img_hr_right = np.concatenate(load_mono_cam(1, 1), axis=-1)
            img_lr_left = np.concatenate(load_mono_cam(0, 4), axis=-1)
            img_lr_right = np.concatenate(load_mono_cam(1, 4), axis=-1)
    
        if self.to_tensor:
            # img_hr_left, img_hr_right, img_lr_left, img_lr_right = augmentation(img_hr_left, img_hr_right, img_lr_left, img_lr_right)
            return toTensor(img_hr_left), toTensor(img_hr_right), toTensor(img_lr_left), toTensor(img_lr_right)
        return img_hr_left, img_hr_right, img_lr_left, img_lr_right

    def __len__(self):
        return len(self.file_list)


def augmentation(hr_image_left, hr_image_right, lr_image_left, lr_image_right):

        if random.random() < 0.5:     # flip horizonly
            lr_image_left_ = lr_image_right[:, ::-1, :]
            lr_image_right_ = lr_image_left[:, ::-1, :]
            hr_image_left_ = hr_image_right[:, ::-1, :]
            hr_image_right_ = hr_image_left[:, ::-1, :]
            lr_image_left, lr_image_right = lr_image_left_, lr_image_right_
            hr_image_left, hr_image_right = hr_image_left_, hr_image_right_

        if random.random() < 0.5:  # flip vertically
            lr_image_left = lr_image_left[::-1, :, :]
            lr_image_right = lr_image_right[::-1, :, :]
            hr_image_left = hr_image_left[::-1, :, :]
            hr_image_right = hr_image_right[::-1, :, :]

        return np.ascontiguousarray(hr_image_left), np.ascontiguousarray(hr_image_right), \
            np.ascontiguousarray(
                lr_image_left), np.ascontiguousarray(lr_image_right)


def toTensor(img):
    if len(img.shape) == 4:
        img = torch.from_numpy(np.transpose(img, (0, 3, 1, 2)))
        if img.shape[1] > 3:
            img = torch.stack([img[:, i]/ (255.0 if i != 3 else 1.0) for i in range(img.shape[1])], dim=1)
        else:
            img = img.float() / 255.0
    else:
        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        if img.shape[0] > 3:
            img = torch.stack([img[i]/ (255.0 if i != 3 else 1.0) for i in range(img.shape[0])], dim=0)
        else:
            img = img.float() / 255.0
    return img

def toNdarray(tensor):
    assert len(tensor.shape) == 4
    ndarray = np.transpose(tensor.data.cpu().numpy(), (0, 2, 3, 1))
    return (ndarray * 255.0).astype('uint8')
