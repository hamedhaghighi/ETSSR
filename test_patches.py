# from torchvision.transforms import ToTensor
import argparse
import os

import tqdm
import yaml
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torch.utils.data import Subset

import dataset
import models.ipassr as ipassr
import models.model as mine
import models.StreoSwinSR as SSR
from dataset import toNdarray, toTensor
from models.model import *
from utils import check_input_size


def patchify_img(img, h_patch, w_patch):
    assert len(img.shape) == 3
    h, w, c = img.shape
    assert h % h_patch == 0 and w % w_patch == 0
    img = img.reshape(h // h_patch, h_patch, w // w_patch, w_patch, c)
    img = img.swapaxes(1, 2)
    return img.reshape(h // h_patch * w // w_patch, h_patch, w_patch, c)


def unify_patches(patches, n_h, n_w):
    b, h, w, c = patches.shape
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
    input_size = tuple([biggest_divisior(cfg.input_resolution[0]),
                       biggest_divisior(cfg.input_resolution[0])])
    input_size = check_input_size(input_size, cfg.w_size)
    net = mine.Net(
        cfg.scale_factor,
        input_size,
        cfg.model,
        IC,
        cfg.w_size,
        cfg.device).to(
        cfg.device) if 'mine' in cfg.model else (
            SSR.Net(
                cfg.scale_factor,
                input_size,
                cfg.model,
                IC,
                cfg.w_size,
                cfg.device).to(
                    cfg.device) if 'swin' in cfg.model else ipassr.Net(
                        cfg.scale_factor,
                        IC).to(
                            cfg.device))
    model_path = os.path.join(cfg.checkpoints_dir,
                              'modelx' + str(cfg.scale_factor) + cfg.ckpt + '.pth')
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
    total_dataset = dataset.DataSetLoader(cfg, to_tensor=False)
    test_set = Subset(
        total_dataset, range(
            len(total_dataset))[
            :len(total_dataset) // 100])
    # test_tq = tqdm.tqdm(total=len(test_set), desc='Iter', position=3)
    psnr_right_list = []
    psnr_left_list = []
    ssim_left_list = []
    ssim_right_list = []
    rand_ind_to_save = np.random.randint(0, len(test_set))
    for idx in tqdm.tqdm(range(len(test_set))):
        HR_left, HR_right, LR_left, LR_right = test_set[idx]
        with torch.no_grad():
            SR_left, SR_right = net(
                toTensor(LR_left).unsqueeze(0).to(
                    cfg.device), toTensor(LR_right).unsqueeze(0).to(
                    cfg.device), is_training=0)
            SR_left, SR_right = torch.clamp(
                SR_left, 0.0, 1.0), torch.clamp(
                SR_right, 0.0, 1.0)
            SR_left, SR_right = toNdarray(SR_left)[0], toNdarray(SR_right)[0]
            psnr_left = compare_psnr(HR_left.astype('uint8'), SR_left)
            psnr_right = compare_psnr(HR_right.astype('uint8'), SR_right)
            ssim_left = compare_ssim(
                HR_left.astype('uint8'), SR_left, multichannel=True)
            ssim_right = compare_ssim(
                HR_right.astype('uint8'), SR_right, multichannel=True)
            psnr_left_list.append(psnr_left)
            psnr_right_list.append(psnr_right)
            ssim_left_list.append(ssim_left)
            ssim_right_list.append(ssim_right)

    print(
        'psnr_left:%.3f' %
        np.array(psnr_left).mean(),
        'psnr_right:%.3f' %
        np.array(psnr_right_list).mean())
    print(
        'ssim_left:%.3f' %
        np.array(ssim_left_list).mean(),
        'ssim_right:%.3f' %
        np.array(ssim_right_list).mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='Path of the config file')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='',
        help='Path of the dataset')
    parser.add_argument(
        '--checkpoints_dir',
        type=str,
        default='',
        help='Path of the dataset')
    parser.add_argument('--fast_test', default=False, action='store_true')

    args = parser.parse_args()
    cfg = cfg_parser(args)
    test(cfg)
    print('Finished!')
