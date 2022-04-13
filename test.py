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
from skimage.measure import compare_psnr, compare_ssim
from utils import check_input_size
import matplotlib.pyplot as plt
import yaml
import dataset
import tqdm

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
    input_size = tuple([biggest_divisior(cfg.input_resolution[0]), biggest_divisior(cfg.input_resolution[0])])
    input_size = check_input_size(input_size, cfg.w_size)
    net = mine.Net(cfg.scale_factor, input_size, cfg.model, IC, cfg.w_size, cfg.device).to(cfg.device) if 'mine' in cfg.model\
        else (SSR.Net(cfg.scale_factor, input_size, cfg.model, IC, cfg.w_size, cfg.device).to(cfg.device) if 'swin' in cfg.model else ipassr.Net(cfg.scale_factor, IC).to(cfg.device))
    model_path = os.path.join(cfg.checkpoints_dir, 'modelx' + str(cfg.scale_factor) + '.pth')
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

    for env in os.listdir(root_dir):
        cfg.data_dir = os.path.join(root_dir, env)
        total_dataset = dataset.DataSetLoader(cfg, to_tensor=False)
        test_set = Subset(total_dataset, range(len(total_dataset))[-len(total_dataset)//10:])
        test_tq = tqdm.tqdm(total=len(test_set), desc='Iter', position=3)
        psnr_right_list=[]
        psnr_left_list=[]
        ssim_left_list =[]
        ssim_right_list=[]
        rand_ind_to_save = np.random.randint(0, len(test_tq))
        for idx in range(len(test_tq)):
            HR_left, HR_right, LR_left, LR_right = test_set[idx]
            h, w, _ = LR_left.shape
            h_patch = biggest_divisior(h)
            w_patch = biggest_divisior(w)
            n_h , n_w = h // h_patch, w // w_patch
            lr_left_patches = patchify_img(LR_left, h_patch, w_patch)
            lr_right_patches = patchify_img(LR_right, h_patch, w_patch)
            # batch_size = lr_left_patches.shape[0]
            batch_size = 2
            sr_left_list = []
            sr_right_list = []
            assert lr_left_patches.shape[0]%batch_size == 0
            for i in range(lr_left_patches.shape[0]//batch_size):
                s = i * batch_size
                e = (i+1) * batch_size
                lr_left_patches_b, lr_right_patches_b = toTensor(lr_left_patches[s:e]), toTensor(lr_right_patches[s:e])
                lr_left_patches_b, lr_right_patches_b = lr_left_patches_b.to(cfg.device), lr_right_patches_b.to(cfg.device)
                with torch.no_grad():
                    SR_left_patches_b, SR_right_patches_b = net(lr_left_patches_b, lr_right_patches_b, is_training=0)
                    SR_left_patches_b, SR_right_patches_b = torch.clamp(SR_left_patches_b, 0, 1), torch.clamp(SR_right_patches_b, 0, 1)
                sr_left_list.append(toNdarray(SR_left_patches_b))
                sr_right_list.append(toNdarray(SR_right_patches_b))
            sr_left_patches = np.concatenate(sr_left_list, axis=0)
            sr_right_patches = np.concatenate(sr_right_list, axis=0)

            sr_left, sr_right = unify_patches(sr_left_patches, n_h, n_w), unify_patches(sr_right_patches, n_h, n_w)
            psnr_left = compare_psnr(HR_left.astype('uint8'), sr_left)
            psnr_right = compare_psnr(HR_right.astype('uint8'), sr_right)
            ssim_left = compare_ssim(HR_left.astype('uint8'), sr_left, multichannel=True)
            ssim_right = compare_ssim(HR_right.astype('uint8'), sr_right, multichannel=True)
            psnr_left_list.append(psnr_left)
            psnr_right_list.append(psnr_right)
            ssim_left_list.append(ssim_left)
            ssim_right_list.append(ssim_right)
            if idx == rand_ind_to_save:
                
                def save_array(array, name):
                    im =Image.fromarray(array)
                    img_path = os.path.join(results_dir,'{}_{}_img_{}.png'.format(name, env, idx))
                    im.save(img_path)

                save_array(sr_left[..., :3].astype('uint8'), 'sr_left')
                save_array(sr_right[..., :3].astype('uint8'), 'sr_right')
                save_array(LR_left[..., :3].astype('uint8'), 'lr_left')
                save_array(LR_right[..., :3].astype('uint8'), 'lr_right')
                save_array(HR_left[..., :3].astype('uint8'), 'hr_left')
                save_array(HR_right[..., :3].astype('uint8'), 'hr_right')
            # plt.figure(0)
            # plt.imshow(sr_left[..., :3].astype('uint8'))
            # plt.figure(1)
            # plt.imshow(sr_right[..., :3].astype('uint8'))
            # plt.show()
            # test_tq.update(1)

        print('env: ', env, 'psnr_left:', np.array(psnr_left_list).mean(), 'psnr_right:', np.array(psnr_right_list).mean())
        print('env: ', env, 'ssim_left:', np.array(ssim_left_list).mean(), 'ssim_right:', np.array(ssim_right_list).mean())

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
    parser.add_argument('--data_dir', type=str, default='', help='Path of the dataset')
    parser.add_argument('--checkpoints_dir', type=str, default='', help='Path of the dataset')
    parser.add_argument('--fast_test', default=False, action='store_true')

    args = parser.parse_args()
    cfg = cfg_parser(args)
    test(cfg)
    print('Finished!')
