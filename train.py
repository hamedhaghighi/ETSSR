import argparse
import os
import random
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import tqdm
import yaml
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torch.utils.data import DataLoader, Subset

import dataset
from dataset import toNdarray, toTensor
from model_selection import model_selection
from utils import check_input_size
from visualizer import Logger


def modify_opt_for_fast_test(opt):
    opt.n_epochs = 2
    opt.batch_size = 1
    opt.exp_name = 'test'
    opt.val_split_ratio = 0.5


class cfg_parser():
    def __init__(self, args):
        opt_dict = yaml.safe_load(open(args.cfg, 'r'))
        for k, v in opt_dict.items():
            setattr(self, k, v)
        if args.data_dir != '':
            self.data_dir = args.data_dir
        if args.test_data_dir != '':
            self.test_data_dir = args.test_data_dir
        self.fast_test = args.fast_test
        self.cfg_path = args.cfg


def step(net, dl, optimizer, vis, idx_epoch, idx_step, cfg, phase):
    dl_iter = iter(dl)
    n_batch = len(dl) if not cfg.fast_test else 2
    loss_list = defaultdict(list)
    tq = tqdm.tqdm(total=n_batch, desc='Iter', position=3)
    avg_sr_loss = None
    net.train(True) if phase == 'train' else net.train(False)
    for _ in range(len(tq)):
        HR_left, HR_right, LR_left, LR_right = next(dl_iter)
        HR_left, HR_right, LR_left, LR_right = HR_left.to(
            cfg.device), HR_right.to(
            cfg.device), LR_left.to(
            cfg.device), LR_right.to(
                cfg.device)
        # check_disparity(LR_left.cpu(), LR_right.cpu())
        if phase == 'train':
            loss = net.calc_loss(LR_left, LR_right, HR_left, HR_right, cfg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                loss = net.calc_loss(LR_left, LR_right, HR_left, HR_right, cfg)
        for k, v in net.get_losses().items():
            loss_list[k].append(v)
        if phase == 'train' and idx_step % 50 == 0:
            avg_loss_list = {k: np.array(v).mean()
                             for k, v in loss_list.items()}
            vis.print_current_losses(
                phase, idx_epoch, idx_step, avg_loss_list, tq)
            vis.plot_current_losses(phase, idx_epoch, avg_loss_list, idx_step)
            loss_list = defaultdict(list)
        tq.update(1)
        idx_step = idx_step + 1 if phase == 'train' else idx_step

    if phase == 'val':
        avg_psnr_left_list = []
        avg_psnr_right_list = []
        avg_ssim_left_list = []
        avg_ssim_right_list = []
        message = ''
        for env in sorted(os.listdir(cfg.test_data_dir)):
            t_cfg = deepcopy(cfg)
            t_cfg.test_data_dir = os.path.join(cfg.test_data_dir, env)
            total_dataset = dataset.DataSetLoader(
                t_cfg, to_tensor=False, test_for_train=True)
            test_set = Subset(total_dataset, range(
                len(total_dataset))[-len(total_dataset) // 10:])
            psnr_right_list = []
            psnr_left_list = []
            ssim_left_list = []
            ssim_right_list = []
            for idx in range(len(test_set)):
                HR_left, HR_right, LR_left, LR_right = test_set[idx]
                # HR_left, HR_right, LR_left, LR_right = HR_left[:120,:360], HR_right[:120,:360], LR_left[:30,:90], LR_right[:30,:90]
                h, w, _ = LR_left.shape
                batch_size = 1
                lr_left, lr_right = toTensor(LR_left).to(
                    cfg.device).unsqueeze(0), toTensor(LR_right).to(
                    cfg.device).unsqueeze(0)
                with torch.no_grad():
                    SR_left, _, _, SR_right, _, _ = net(lr_left, lr_right)
                SR_left, SR_right = toNdarray(
                    torch.clamp(
                        SR_left, 0, 1)).squeeze(), toNdarray(
                    torch.clamp(
                        SR_right, 0, 1)).squeeze()

                psnr_left = compare_psnr(
                    HR_left[..., :3].astype('uint8'), SR_left)
                psnr_right = compare_psnr(
                    HR_right[..., :3].astype('uint8'), SR_right)
                ssim_left = compare_ssim(HR_left[..., :3].astype(
                    'uint8'), SR_left, multichannel=True)
                ssim_right = compare_ssim(HR_right[..., :3].astype(
                    'uint8'), SR_right, multichannel=True)
                psnr_left_list.append(psnr_left)
                psnr_right_list.append(psnr_right)
                ssim_left_list.append(ssim_left)
                ssim_right_list.append(ssim_right)
            message += 'env: ' + env + ' psnr_left:%.3f' % np.array(psnr_left_list).mean(
            ) + ' psnr_right:%.3f' % np.array(psnr_right_list).mean() + '\n'
            message += 'env: ' + env + ' ssim_left:%.3f' % np.array(ssim_left_list).mean(
            ) + ' ssim_right:%.3f' % np.array(ssim_right_list).mean() + '\n'
            avg_psnr_left_list.extend(psnr_left_list)
            avg_psnr_right_list.extend(psnr_right_list)
            avg_ssim_left_list.extend(ssim_left_list)
            avg_ssim_right_list.extend(ssim_right_list)

        avg_loss_list = {k: np.array(v).mean() for k, v in loss_list.items()}
        avg_loss_list['psnr_left'] = np.array(avg_psnr_left_list).mean()
        avg_loss_list['psnr_right'] = np.array(avg_psnr_right_list).mean()
        avg_loss_list['ssim_left'] = np.array(avg_ssim_left_list).mean()
        avg_loss_list['ssim_right'] = np.array(avg_ssim_right_list).mean()

        avg_sr_loss = avg_loss_list['SR']
        vis.print_current_losses(phase, idx_epoch, idx_step, avg_loss_list, tq)
        vis.plot_current_losses(phase, idx_epoch, avg_loss_list, idx_step)
        log_name = os.path.join(
            cfg.checkpoints_dir,
            cfg.exp_name,
            'loss_log.txt')
        with open(log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    return idx_step, avg_sr_loss


def train(train_loader, val_loader, cfg):
    IC = cfg.input_channel
    input_size = check_input_size(cfg.input_resolution, cfg.w_size)
    net = model_selection(
        cfg.model,
        cfg.scale_factor,
        input_size[0],
        input_size[1],
        IC,
        cfg.w_size,
        cfg.device)
    if cfg.load:
        model_path = os.path.join(cfg.checkpoints_dir,
                                  cfg.exp_name,
                                  'modelx' + str(cfg.scale_factor) + '_last' + '.pth')
        if os.path.isfile(model_path):
            model = torch.load(model_path, map_location={'cuda:0': cfg.device})
            net.load_state_dict(model['state_dict'])
            cfg.start_epoch = model["epoch"]
            cfg.lr = cfg.lr * cfg.gamma ** (model["epoch"] // cfg.n_steps)
        else:
            print("=> no model found at '{}'".format(cfg.load_model))

    # net = torch.nn.DataParallel(net, device_ids=[0, 1])
    optimizer = torch.optim.Adam(
        [paras for paras in net.parameters() if paras.requires_grad], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)
    min_val_loss = np.inf
    idx_step = cfg.start_epoch * len(train_loader)
    best_ckpt_path = os.path.join(cfg.checkpoints_dir,
                                  cfg.exp_name,
                                  'modelx' + str(cfg.scale_factor) + '_best' + '.pth')
    last_ckpt_path = os.path.join(cfg.checkpoints_dir,
                                  cfg.exp_name,
                                  'modelx' + str(cfg.scale_factor) + '_last' + '.pth')
    vis = Logger(cfg)
    for idx_epoch in range(cfg.start_epoch, cfg.n_epochs):
        idx_step, _ = step(net, train_loader, optimizer, vis,
                           idx_epoch, idx_step, cfg, 'train')
        _, val_loss = step(net, val_loader, optimizer, vis,
                           idx_epoch, idx_step, cfg, 'val')
        scheduler.step()
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save({'epoch': idx_epoch + 1,
                        'state_dict': net.state_dict()},
                       best_ckpt_path)
        torch.save({'epoch': idx_epoch + 1,
                    'state_dict': net.state_dict()},
                   last_ckpt_path)


def main(cfg):
    total_set = dataset.DataSetLoader(cfg, max_data_size=cfg.max_data_size)
    total_indices = list(range(len(total_set)))
    random.shuffle(total_indices)
    train_indcs = total_indices[int(cfg.val_split_ratio * len(total_set)):]
    val_indcs = total_indices[:int(cfg.val_split_ratio * len(total_set))]
    train_dataset = Subset(total_set, train_indcs)
    val_dataset = Subset(total_set, val_indcs)
    train_loader = DataLoader(
        dataset=train_dataset,
        num_workers=2,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True)
    val_loader = DataLoader(
        dataset=val_dataset,
        num_workers=2,
        batch_size=cfg.batch_size,
        shuffle=False)
    train(train_loader, val_loader, cfg)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='Path of the config file')
    parser.add_argument(
        '--test_data_dir',
        type=str,
        default='',
        help='directory of test dataset')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='',
        help='Path of the dataset')
    parser.add_argument('--fast_test', default=False, action='store_true', help='if true to training will be run for few epochs to make sure it works')
    args = parser.parse_args()
    cfg = cfg_parser(args)
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if cfg.fast_test:
        modify_opt_for_fast_test(cfg)
    main(cfg)
