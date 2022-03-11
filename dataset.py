from PIL import Image
import os
from torch.utils.data.dataset import Dataset
import random
import torch
import numpy as np


class DataSetLoader(Dataset):
    def __init__(self, cfg, to_tensor=True):
        super(DataSetLoader, self).__init__()
        self.cfg = cfg
        self.dataset_dir = cfg.data_dir
        self.file_list = sorted(os.listdir(self.dataset_dir))
        self.to_tensor = to_tensor

    def read_img(self, img_path):
        ext = img_path.split('.')[-1]
        if ext == 'npy':
            return np.load(img_path)[..., :self.cfg.input_channel]
        elif ext == 'png':
            assert self.cfg.input_channel == 3
            img = Image.open(img_path)
            return np.array(img,  dtype=np.float32)
        

    def __getitem__(self, index):
        ext = '.npy' if self.cfg.train_on_sim else '.png'
        img_hr_left = self.read_img(self.dataset_dir + '/' + self.file_list[index] + '/hr0' + ext)[..., :3]
        img_hr_right = self.read_img(self.dataset_dir + '/' + self.file_list[index] + '/hr1' + ext)[..., :3]
        img_lr_left = self.read_img(self.dataset_dir + '/' + self.file_list[index] + '/lr0' + ext)
        img_lr_right = self.read_img(self.dataset_dir + '/' + self.file_list[index] + '/lr1' + ext)
        # TODO: change this in the dataset
        img_lr_left[..., 3] = img_lr_left[..., 3] / 2.0
        img_lr_right[..., 3] = img_lr_right[..., 3] / 2.0
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
