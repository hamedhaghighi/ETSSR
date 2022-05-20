import dataset
from dataset import toNdarray, toTensor
from timm.models.layers import trunc_normal_
from models.SwinTransformer import SwinAttn
from models.CoSwinTransformer import CoSwinAttn
from utils import disparity_alignment
import os
from torch.utils.data import Subset
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BaseModel(nn.Module):
    def __init__(self, upscale_factor, img_size, model, input_channel=3, w_size=8, device='cpu'):
        super(BaseModel, self).__init__()
        self.loss_names = ['SR']
        
    def get_losses(self):
        loss_dict = {k: getattr(self, 'loss_' + k).data.cpu() for k in self.loss_names}
        return loss_dict

    def calc_loss(self, LR_left, LR_right, HR_left, HR_right, cfg, is_train=True):

        criterion_L1 = torch.nn.L1Loss().to(cfg.device)
        SR_left, SR_right = self.forward(LR_left, LR_right)
        ''' SR Loss '''
        self.loss_SR = criterion_L1(SR_left, HR_left) + criterion_L1(SR_right, HR_right)
        # if not is_train:
        #     self.loss_names.extend(['psnr_left', 'ssim_left', 'psnr_right', 'ssim_right'])
        #     nd_array_sr_left, nd_array_sr_right = toNdarray(SR_left), toNdarray(SR_right)
        #     nd_array_hr_left, nd_array_hr_right = toNdarray(HR_left), toNdarray(HR_right)
        #     psnr_left = [compare_psnr(hr, sr) for hr, sr in zip(nd_array_hr_left, nd_array_sr_left)]
        #     psnr_right = [compare_psnr(hr, sr) for hr, sr in zip(nd_array_hr_right, nd_array_sr_right)]
        #     ssim_left = [compare_ssim(hr, sr, multichannel=True) for hr, sr in zip(nd_array_hr_left, nd_array_sr_left)]
        #     ssim_right = [compare_ssim(hr, sr, multichannel=True) for hr, sr in zip(nd_array_hr_right, nd_array_sr_right)]
        #     self.loss_psnr_left = torch.tensor(np.array(psnr_left).mean())
        #     self.loss_psnr_right = torch.tensor(np.array(psnr_right).mean())
        #     self.loss_ssim_left = torch.tensor(np.array(ssim_left).mean())
        #     self.loss_ssim_right = torch.tensor(np.array(ssim_right).mean())
    
        return self.loss_SR

    def flop(self, H, W):
        return 0

    




