from itertools import dropwhile
from unittest.mock import patch
from matplotlib.pyplot import axis
from torch.autograd import Variable
from PIL import Image
# from torchvision.transforms import ToTensor
import argparse
import os
from model import *
from utils import toNdarray, toTensor
from skimage.measure import compare_psnr, compare_ssim
import matplotlib.pyplot as plt
import cv2
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testset_dir', type=str, default='/media/oem/Local Disk/Phd-datasets/iPASSR/datasets_Airsim')
    parser.add_argument('--scale_factor', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_name', type=str, default='iPASSR_2xSR')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    parser.add_argument('--exp_name', type=str, default='orig')
    return parser.parse_args()


def patchify_img(img, h_patch, w_patch):
    assert len(img.shape) == 3
    h ,w, c = img.shape
    assert h%h_patch == 0 and w%w_patch  == 0
    img = img.reshape(h//h_patch, h_patch, w//w_patch, w_patch, c)
    img = img.swapaxes(1, 2)
    return img.reshape(h//h_patch * w//w_patch, h_patch, w_patch, c)

def unify_patches(patches, n_h, n_w):
    b , h, w, c = patches.shape
    patches = patches.reshape(n_h, n_w, h, w, c)
    patches = patches.swapaxes(1, 2)
    return patches.reshape(h * n_h, w * n_w, c)

def biggest_divisior(n):
    for i in range(100, 10, -1):
        if n % i == 0:
            return i

def downsample(img):
    img = cv2.resize(img.astype('uint8'), (int(0.5 * img.shape[1]), int(0.5 * img.shape[0])), interpolation=cv2.INTER_CUBIC)
    img = img.astype('float32')
    # img[:, :, 3] = cv2.resize(img[:, :, 3], (int(0.5 * img.shape[1]), int(0.5 * img.shape[0])))
    return img

def test(cfg):
    net = Net(cfg.scale_factor).to(cfg.device)
    model_path = os.path.join(
    cfg.checkpoints_dir, cfg.exp_name, 'modelx' + str(cfg.scale_factor) + '.pth')
    model = torch.load(model_path, map_location={'cuda:0': cfg.device})
    net.load_state_dict(model['state_dict'])
    folder_list = os.listdir(cfg.testset_dir)
    for idx, folder in enumerate(folder_list):
        file_path = os.path.join(cfg.testset_dir, folder,'im0.npy')
        HR_left = np.load(file_path)[...,:3].astype('uint8')
        file_path = os.path.join(cfg.testset_dir, folder, 'im1.npy')
        HR_right = np.load(file_path)[..., :3].astype('uint8')
        LR_left = downsample(HR_left)
        LR_right = downsample(HR_right)
        h, w, _ = LR_left.shape
        h_patch = biggest_divisior(h)
        w_patch = biggest_divisior(w)
        
        n_h , n_w = h // h_patch, w // w_patch
        lr_left_patches = patchify_img(LR_left, h_patch, w_patch)
        lr_right_patches = patchify_img(LR_right, h_patch, w_patch)
        batch_size = lr_left_patches.shape[0]
        sr_left_list = []
        sr_right_list = []
        assert lr_left_patches.shape[0]%batch_size == 0
        for i in range(lr_left_patches.shape[0]//batch_size):
            s = i * batch_size
            e = (i+1) * batch_size
            lr_left_patches_b, lr_right_patches_b = toTensor(lr_left_patches[s:e]), toTensor(lr_right_patches[s:e])
            lr_left_patches_b, lr_right_patches_b = lr_left_patches_b.to(cfg.device), lr_right_patches_b.to(cfg.device)
            with torch.no_grad():
                start = time.time()
                import pdb; pdb.set_trace()
                SR_left_patches_b, SR_right_patches_b = net(lr_left_patches_b, lr_right_patches_b, is_training=0)
                print(time.time() - start)
                SR_left_patches_b, SR_right_patches_b = torch.clamp(SR_left_patches_b, 0, 1), torch.clamp(SR_right_patches_b, 0, 1)
            sr_left_list.append(toNdarray(SR_left_patches_b))
            sr_right_list.append(toNdarray(SR_right_patches_b))
        sr_left_patches = np.concatenate(sr_left_list, axis=0)
        sr_right_patches = np.concatenate(sr_right_list, axis=0)
        sr_left, sr_right = unify_patches(sr_left_patches, n_h, n_w), unify_patches(sr_right_patches, n_h, n_w)
        psnr_left = compare_psnr(np.asarray(HR_left), sr_left)
        psnr_right = compare_psnr(np.asarray(HR_right), sr_right)
        ssim_left = compare_ssim(np.asarray(HR_left), sr_left, multichannel=True)
        ssim_right = compare_ssim(np.asarray(HR_right), sr_right, multichannel=True)
        
        print('id:', idx, 'psnr_left:', psnr_left, 'psnr_right:', psnr_right)
        print('id:', idx, 'ssim_left:', ssim_left, 'ssim_right:', ssim_right)
        plt.figure(0)
        plt.imshow(sr_right)
        plt.figure(1)
        plt.imshow(HR_right)
        plt.show()
        # save_path = './results/' + cfg.model_name + '/' + cfg.dataset
            # if not os.path.exists(save_path):
            #     os.makedirs(save_path)
            # SR_left_img = transforms.ToPILImage()(torch.squeeze(SR_left.data.cpu(), 0))
            # SR_left_img.save(save_path + '/' + scene_name + '_L.png')
            # SR_right_img = transforms.ToPILImage()(torch.squeeze(SR_right.data.cpu(), 0))
            # SR_right_img.save(save_path + '/' + scene_name + '_R.png')


if __name__ == '__main__':
    cfg = parse_args()
    test(cfg)
    print('Finished!')
