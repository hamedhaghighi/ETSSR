from unittest.mock import patch
from matplotlib.pyplot import axis
from torch.autograd import Variable
from torch.utils.data import Subset
import models.ipassr as ipassr
import models.model as mine
import models.StreoSwinSR as SSR
from PIL import Image
# from torchvision.transforms import ToTensor
import argparse
import os
from models.model import *
from dataset import toNdarray, toTensor
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from utils import check_input_size
import matplotlib.pyplot as plt
import yaml
import dataset
import tqdm
import cv2


def patchify_img(img, h_patch, w_patch):
    assert len(img.shape) == 3
    h, w, c = img.shape
    assert h % h_patch == 0 and w % w_patch == 0
    img = img.reshape(h//h_patch, h_patch, w//w_patch, w_patch, c)
    img = img.swapaxes(1, 2)
    return img.reshape(h//h_patch * w//w_patch, h_patch, w_patch, c)


def unify_patches(patches, n_h, n_w):
    b, h, w, c = patches.shape
    patches = patches.reshape(n_h, n_w, h, w, c)
    patches = patches.swapaxes(1, 2)
    return patches.reshape(h * n_h, w * n_w, c)


def biggest_divisior(n):
    for i in range(100, 10, -1):
        if n % i == 0:
            return i


def _pad(img, pad_h, pad_w):
    return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)))


def calc_metrics(sr_left, sr_right, hr_left, hr_right, psnr_left_list, psnr_right_list, ssim_left_list, ssim_right_list):
    psnr_left = compare_psnr(hr_left.astype('uint8'), sr_left)
    psnr_right = compare_psnr(hr_right.astype('uint8'), sr_right)
    ssim_left = compare_ssim(hr_left.astype('uint8'), sr_left, multichannel=True)
    ssim_right = compare_ssim(hr_right.astype('uint8'), sr_right, multichannel=True)
    psnr_left_list.append(psnr_left)
    psnr_right_list.append(psnr_right)
    ssim_left_list.append(ssim_left)
    ssim_right_list.append(ssim_right)
    return psnr_left_list, psnr_right_list, ssim_left_list, ssim_right_list

class cfg_parser():
    def __init__(self, args):
        opt_dict = yaml.safe_load(open(args.cfg, 'r'))
        for k, v in opt_dict.items():
            setattr(self, k, v)
        if args.data_dir != '':
            self.data_dir = args.data_dir
        if args.checkpoints_dir != '':
            self.checkpoints_dir = args.checkpoints_dir
        self.fast_test = args.fast_test
        self.cfg_path = args.cfg


def test(cfg):
    IC = cfg.input_channel
    input_size = tuple([biggest_divisior(cfg.input_resolution[0]),
                        biggest_divisior(cfg.input_resolution[0])])
    input_size = check_input_size(input_size, cfg.w_size)
    net = mine.Net(cfg.scale_factor, input_size, cfg.model, IC, cfg.w_size, cfg.device).to(cfg.device) if 'mine' in cfg.model\
        else (SSR.Net(cfg.scale_factor, input_size, cfg.model, IC, cfg.w_size, cfg.device).to(cfg.device) if 'swin' in cfg.model else ipassr.Net(cfg.scale_factor, IC).to(cfg.device))
    model_path = os.path.join(
        cfg.checkpoints_dir, 'modelx' + str(cfg.scale_factor) + cfg.ckpt + '.pth')
    model = torch.load(model_path, map_location={'cuda:0': cfg.device})
    model_state_dict = dict()
    for k, v in model['state_dict'].items():
        if 'attn_mask' not in k:
            model_state_dict[k] = v
    net.load_state_dict(model_state_dict)
    image_folders = os.listdir()
    root_dir = cfg.data_dir
    results_dir = os.path.join(cfg.checkpoints_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    avg_psnr_left_list = []
    avg_psnr_right_list = []
    avg_ssim_left_list = []
    avg_ssim_right_list = []
    psnr_right_list_bcb, psnr_left_list_bcb, ssim_left_list_bcb, ssim_right_list_bcb = [], [], [], []
    indices_to_save = {'Town02': [1, 2], 'Town07': [1, 4]}
    with torch.no_grad():
        for env in sorted(os.listdir(root_dir)):
            cfg.data_dir = os.path.join(root_dir, env)
            total_dataset = dataset.DataSetLoader(cfg, to_tensor=False)
            test_set = Subset(total_dataset, range(
                len(total_dataset))[-len(total_dataset)//10:])
            # test_tq = tqdm.tqdm(total=len(test_set), desc='Iter', position=3)
            psnr_right_list, psnr_left_list, ssim_left_list, ssim_right_list = [] , [], [], []
            
            for idx in range(len(test_set)):
                HR_left, HR_right, LR_left, LR_right = test_set[idx]
                h, w, _ = LR_left.shape
                h_patch = biggest_divisior(h)
                w_patch = biggest_divisior(w)
                h_patch = 360
                w_patch = 640
                pad_h, pad_w = (h_patch - (h % h_patch)) % h_patch, (w_patch - (w % w_patch)) % w_patch
                LR_left, LR_right = _pad(LR_left, pad_h, pad_w), _pad(
                    LR_right, pad_h, pad_w)
                h, w, _ = LR_left.shape
                n_h, n_w = h // h_patch, w // w_patch
                lr_left_patches = patchify_img(LR_left, h_patch, w_patch)
                lr_right_patches = patchify_img(LR_right, h_patch, w_patch)
                if cfg.local_metric:
                    HR_left, HR_right = _pad(HR_left, cfg.scale_factor * pad_h, cfg.scale_factor * pad_w), _pad(
                        HR_right, cfg.scale_factor * pad_h, cfg.scale_factor * pad_w)
                    hr_left_patches = patchify_img(HR_left, cfg.scale_factor * h_patch, cfg.scale_factor * w_patch)
                    hr_right_patches = patchify_img(HR_right, cfg.scale_factor * h_patch, cfg.scale_factor * w_patch)
                # batch_size = lr_left_patches.shape[0]
                batch_size = 2 if cfg.batch_size != -1 else lr_left_patches.shape[0]
                sr_left_list = []
                sr_right_list = []
                assert lr_left_patches.shape[0] % batch_size == 0
                for i in range(lr_left_patches.shape[0]//batch_size):
                    s = i * batch_size
                    e = (i+1) * batch_size
                    lr_left_patches_b, lr_right_patches_b = toTensor(
                        lr_left_patches[s:e]), toTensor(lr_right_patches[s:e])
                    lr_left_patches_b, lr_right_patches_b = lr_left_patches_b.to(
                        cfg.device), lr_right_patches_b.to(cfg.device)
                    SR_left_patches_b, SR_right_patches_b = net(
                        lr_left_patches_b, lr_right_patches_b, is_training=0)
                    SR_left_patches_b, SR_right_patches_b = torch.clamp(
                        SR_left_patches_b, 0, 1), torch.clamp(SR_right_patches_b, 0, 1)
                    sr_left_list.append(toNdarray(SR_left_patches_b))
                    sr_right_list.append(toNdarray(SR_right_patches_b))
                sr_left_patches = np.concatenate(sr_left_list, axis=0)
                sr_right_patches = np.concatenate(sr_right_list, axis=0)
                if cfg.local_metric:
                    for i in range(sr_left_patches.shape[0]):
                        psnr_left_list, psnr_right_list, ssim_left_list, ssim_right_list = calc_metrics(
                            sr_left_patches[i], sr_right_patches[i], hr_left_patches[i], hr_right_patches[i], psnr_left_list, psnr_right_list, ssim_left_list, ssim_right_list)

                sr_left, sr_right = unify_patches(sr_left_patches, n_h, n_w), unify_patches(sr_right_patches, n_h, n_w)
                sr_left, sr_right = sr_left[:, :cfg.scale_factor * cfg.input_resolution[1]], sr_right[:, :cfg.scale_factor * cfg.input_resolution[1]]
                if not cfg.local_metric:
                    dst_shape = (LR_left.shape[1] * cfg.scale_factor, LR_left.shape[0] * cfg.scale_factor)
                    sr_left_bcb, sr_right_bcb = cv2.resize(LR_left[..., :3].astype('uint8'), dst_shape, interpolation=cv2.INTER_CUBIC), cv2.resize(
                        LR_right[..., :3].astype('uint8'), dst_shape, interpolation=cv2.INTER_CUBIC)
                    psnr_left_list, psnr_right_list, ssim_left_list, ssim_right_list = calc_metrics(
                        sr_left, sr_right, HR_left[..., :3], HR_right[..., :3], psnr_left_list, psnr_right_list, ssim_left_list, ssim_right_list)
                    psnr_left_list_bcb, psnr_right_list_bcb, ssim_left_list_bcb, ssim_right_list_bcb = calc_metrics(
                        sr_left_bcb, sr_right_bcb, HR_left[..., :3], HR_right[..., :3], psnr_left_list_bcb, psnr_right_list_bcb, ssim_left_list_bcb, ssim_right_list_bcb)
                    


                if env in indices_to_save and idx in indices_to_save[env]:
                    def save_array(array, name):
                        im = Image.fromarray(array)
                        img_path = os.path.join(
                            results_dir, '{}_{}_img_{}.png'.format(name, env, idx))
                        im.save(img_path)
                    save_array(sr_left[..., :3].astype('uint8'), 'sr_left')
                    save_array(sr_right[..., :3].astype('uint8'), 'sr_right')
                    save_array(LR_left[..., :3].astype('uint8'), 'lr_left')
                    save_array(LR_right[..., :3].astype('uint8'), 'lr_right')
                    save_array(HR_left[..., :3].astype('uint8'), 'hr_left')
                    save_array(HR_right[..., :3].astype('uint8'), 'hr_right')

            print('env: ', env, 'psnr_left:%.3f' % np.array(psnr_left_list).mean(), 'psnr_right:%.3f' % np.array(psnr_right_list).mean())
            print('env: ', env, 'ssim_left:%.3f' % np.array(ssim_left_list).mean(), 'ssim_right:%.3f' % np.array(ssim_right_list).mean())
            avg_psnr_left_list.extend(psnr_left_list)
            avg_psnr_right_list.extend(psnr_right_list)
            avg_ssim_left_list.extend(ssim_left_list)
            avg_ssim_right_list.extend(ssim_right_list)

    print('psnr_left:%.3f' % np.array(avg_psnr_left_list).mean(),
          'psnr_right:%.3f' % np.array(avg_psnr_right_list).mean())
    print('ssim_left:%.3f' % np.array(avg_ssim_left_list).mean(),
          'ssim_right:%.3f' % np.array(avg_ssim_right_list).mean())
    print('psnr_left_bcb:%.3f' % np.array(psnr_left_list_bcb).mean(),
          'psnr_right_bcb:%.3f' % np.array(psnr_right_list_bcb).mean())
    print('ssim_left_bcb:%.3f' % np.array(ssim_left_list_bcb).mean(),
          'ssim_right_bcb:%.3f' % np.array(ssim_right_list_bcb).mean())
    # save_path = './results/' + cfg.model_name + '/' + cfg.dataset
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # SR_left_img = transforms.ToPILImage()(torch.squeeze(SR_left.data.cpu(), 0))
    # SR_left_img.save(save_path + '/' + scene_name + '_L.png')
    # SR_right_img = transforms.ToPILImage()(torch.squeeze(SR_right.data.cpu(), 0))
    # SR_right_img.save(save_path + '/' + scene_name + '_R.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='Path of the config file')
    parser.add_argument('--data_dir', type=str, default='',
                        help='Path of the dataset')
    parser.add_argument('--checkpoints_dir', type=str,
                        default='', help='Path of the dataset')
    parser.add_argument('--fast_test', default=False, action='store_true')

    args = parser.parse_args()
    cfg = cfg_parser(args)
    test(cfg)
    print('Finished!')
