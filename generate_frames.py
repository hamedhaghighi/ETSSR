# from torchvision.transforms import ToTensor
from model_selection import model_selection
from dataset import toNdarray, toTensor
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from unittest.mock import patch
from matplotlib.pyplot import axis
from torch.autograd import Variable
from torch.utils.data import Subset
import models.ipassr as ipassr
import models.model as mine
import models.StreoSwinSR as SSR
from PIL import Image
import argparse
import os
from models.model import *
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



def _pad(img, pad_h, pad_w):
    return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)))


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
    input_size = tuple([int(cfg.input_resolution[0] * cfg.sample_ratio), int(cfg.input_resolution[1] * cfg.sample_ratio)])
    input_size = check_input_size(input_size, cfg.w_size)

    if not 'bicubic' in cfg.model:
        net = model_selection(cfg.model, cfg.scale_factor, input_size[0], input_size[1], IC, cfg.w_size, cfg.device)
        model_path = os.path.join(cfg.checkpoints_dir, 'modelx' + str(cfg.scale_factor) + cfg.ckpt + '.pth')
        model = torch.load(model_path, map_location={'cuda:0': cfg.device})
        model_state_dict = dict()
        for k, v in model['state_dict'].items():
            if 'attn_mask' not in k:
                model_state_dict[k] = v
        net.load_state_dict(model_state_dict)

    ## Reading dataset  ########################################################################
    root_dir = cfg.data_dir
    results_dir = os.path.join(cfg.checkpoints_dir, 'results_frames')
    os.makedirs(results_dir, exist_ok=True)
    env = 'Town01'
    with torch.no_grad():
        cfg.data_dir = os.path.join(root_dir, env)
        total_dataset = dataset.DataSetLoader(cfg, to_tensor=False)
        # test_set = Subset(total_dataset, range(len(total_dataset))[-len(total_dataset)//10:]) if cfg.metric_for_all else total_dataset
        # test_tq = tqdm.tqdm(total=len(test_set), desc='Iter', position=3)
        for idx in range(len(total_dataset)):
            data_idx = int(total_dataset.file_list[idx].split('_')[-1])
            HR_left, HR_right, LR_left, LR_right = total_dataset[idx]
            h, w, _ = LR_left.shape
            
            h_patch = h
            w_patch = w
            pad_h, pad_w = (h_patch - (h % h_patch)) % h_patch, (w_patch - (w % w_patch)) % w_patch
            LR_left, LR_right = _pad(LR_left, pad_h, pad_w), _pad(LR_right, pad_h, pad_w)
            h, w, _ = LR_left.shape
            n_h, n_w = h // h_patch, w // w_patch
            lr_left_patches = patchify_img(LR_left, h_patch, w_patch)

            lr_right_patches = patchify_img(LR_right, h_patch, w_patch)

            # batch_size = lr_left_patches.shape[0]

            ## Feeding to model
            if not 'bicubic' in cfg.model:
                batch_size = 1
                sr_left_list = []
                sr_right_list = []

                assert lr_left_patches.shape[0] % batch_size == 0
                for i in range(lr_left_patches.shape[0]//batch_size):
                    s = i * batch_size
                    e = (i+1) * batch_size
                    lr_left_patches_b, lr_right_patches_b = toTensor(lr_left_patches[s:e]), toTensor(lr_right_patches[s:e])
                    lr_left_patches_b, lr_right_patches_b = lr_left_patches_b.to(cfg.device), lr_right_patches_b.to(cfg.device)
                    SR_left_patches_b, SR_right_patches_b = net(lr_left_patches_b, lr_right_patches_b)
                    SR_left_patches_b, SR_right_patches_b = torch.clamp(SR_left_patches_b, 0, 1), torch.clamp(SR_right_patches_b, 0, 1)
                    sr_left_list.append(toNdarray(SR_left_patches_b))
                    sr_right_list.append(toNdarray(SR_right_patches_b))
                sr_left_patches = np.concatenate(sr_left_list, axis=0)
                sr_right_patches = np.concatenate(sr_right_list, axis=0)
                sr_left, sr_right = unify_patches(sr_left_patches, n_h, n_w), unify_patches(sr_right_patches, n_h, n_w)
                sr_left, sr_right = sr_left[:LR_left.shape[0] * cfg.scale_factor, :LR_left.shape[1] * cfg.scale_factor], sr_right[:LR_left.shape[0] * cfg.scale_factor, :LR_left.shape[1] * cfg.scale_factor]
            else:
                dst_shape = (LR_left.shape[1] * cfg.scale_factor, LR_left.shape[0] * cfg.scale_factor)
                sr_left, sr_right = cv2.resize(LR_left[..., :3].astype('uint8'), dst_shape, interpolation=cv2.INTER_CUBIC), cv2.resize(LR_right[..., :3].astype('uint8'), dst_shape, interpolation=cv2.INTER_CUBIC)
                


            def save_array(array, name, psnr=None, ssim=None):
                im = Image.fromarray(array)
                img_path = os.path.join(
                    results_dir, env, name, 'img_{}.png'.format(data_idx))
                im.save(img_path)
                
            save_array(sr_left[..., :3].astype('uint8'), 'left')
            save_array(sr_right[..., :3].astype('uint8'), 'right')
            if cfg.save_hr:
                save_array(HR_left[..., :3].astype('uint8'), 'hr_left')
                save_array(HR_right[..., :3].astype('uint8'), 'hr_right')




if __name__ == '__main__':
    option = 'generate_vid'
    option = 'generate_frames'
    if option == 'generate_frames':
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
    else:    

        image_folder = 'C:/Users/hamed/Desktop/PhD_proj/StereoSR/checkpoints/ETSSR/results_frames'
        video_name = 'video.mp4'

        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        images_indx = np.argsort([int(img.split('_')[4].split('.')[0]) for img in images])
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video = cv2.VideoWriter(os.path.join(image_folder,video_name), fourcc, 3, (width,height))

        for ind in images_indx:
            video.write(cv2.imread(os.path.join(image_folder, images[ind])))

        cv2.destroyAllWindows()
        video.release()

    
