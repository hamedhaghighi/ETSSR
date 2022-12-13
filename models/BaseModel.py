
import torch
import torch.nn as nn



class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.loss_names = ['SR']

    def get_losses(self):
        loss_dict = {k: getattr(self, 'loss_' + k).data.cpu()
                     for k in self.loss_names}
        return loss_dict

    def calc_loss(self, LR_left, LR_right, HR_left, HR_right, cfg):

        criterion_L1 = torch.nn.L1Loss().to(cfg.device)
        SR_left, SR_right = self.forward(LR_left, LR_right)
        ''' SR Loss '''
        self.loss_SR = criterion_L1(
            SR_left, HR_left) + criterion_L1(SR_right, HR_right)
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
