from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch.backends.cudnn as cudnn
import argparse
from utils import *
import models.ipassr as ipassr
import models.model as mine
import models.StreoSwinSR as SSR
import dataset
from visualizer import Logger
from collections import defaultdict
from utils import check_input_size
import torch
import numpy as np
import tqdm
import yaml
import os

def modify_opt_for_fast_test(opt):
    opt.n_epochs = 2
    opt.batch_size = 1
    opt.exp_name = 'test'
    # opt.epoch_decay = opt.n_epochs//2
    # opt.display_freq = 1
    # opt.print_freq = 1
    # opt.save_latest_freq = 100
    # opt.max_dataset_size = 10




class cfg_parser():
    def __init__(self, args):
        opt_dict = yaml.safe_load(open(args.cfg, 'r'))
        for k, v in opt_dict.items():
            setattr(self, k, v)
        if args.data_dir != '':
            self.data_dir = args.data_dir
        self.fast_test = args.fast_test
        self.cfg_path = args.cfg

def step(net, dl, optimizer, vis, idx_epoch, idx_step, cfg, phase):
    dl_iter = iter(dl)
    n_batch = len(dl) if not cfg.fast_test else 2
    loss_list = defaultdict(list)
    tq = tqdm.tqdm(total=n_batch, desc='Iter', position=3)
    for _ in range(len(tq)):
        HR_left, HR_right, LR_left, LR_right = next(dl_iter)
        HR_left, HR_right, LR_left, LR_right = HR_left.to(cfg.device), HR_right.to(cfg.device), LR_left.to(cfg.device), LR_right.to(cfg.device)
        if phase == 'train':
            net.train(True)
            loss = net.calc_loss(LR_left, LR_right, HR_left, HR_right, cfg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            net.train(False)
            with torch.no_grad():
                loss = net.calc_loss(LR_left, LR_right, HR_left, HR_right, cfg)
        for k, v in net.get_losses().items():
            loss_list[k].append(v)
        if phase == 'train' and idx_step % 50 == 0:
            avg_loss_list = {k: np.array(v).mean() for k, v in loss_list.items()}
            vis.print_current_losses(phase, idx_epoch, idx_step, avg_loss_list, tq)
            vis.plot_current_losses(phase, idx_epoch, avg_loss_list, idx_step)
            loss_list = defaultdict(list)
        tq.update(1)
        idx_step = idx_step + 1 if phase == 'train' else idx_step
    
    if phase == 'val':
        avg_loss_list = {k: np.array(v).mean() for k, v in loss_list.items()}
        vis.print_current_losses(phase, idx_epoch, idx_step, avg_loss_list, tq)
        vis.plot_current_losses(phase, idx_epoch, avg_loss_list, idx_step)
        loss_list = defaultdict(list)
    return idx_step

def train(train_loader, val_loader, cfg):
    IC = cfg.input_channel
    input_size = check_input_size(cfg.input_resolution, cfg.w_size)
    net = mine.Net(cfg.scale_factor, input_size, cfg.model, IC, cfg.w_size, cfg.device).to(cfg.device) if 'mine' in cfg.model\
        else (SSR.Net(cfg.scale_factor, input_size, cfg.model, IC, cfg.w_size, cfg.device).to(cfg.device) if 'swin' in cfg.model else ipassr.Net(cfg.scale_factor, IC).to(cfg.device))
    cudnn.benchmark = True
    scale = cfg.scale_factor
    if cfg.load:
        model_path = os.path.join(cfg.checkpoints_dir, cfg.exp_name, 'modelx' + str(cfg.scale_factor) + '.pth')
        if os.path.isfile(model_path):
            model = torch.load(model_path, map_location={'cuda:0': cfg.device})
            net.load_state_dict(model['state_dict'])
            cfg.start_epoch = model["epoch"]
        else:
            print("=> no model found at '{}'".format(cfg.load_model))

    # net = torch.nn.DataParallel(net, device_ids=[0, 1])
    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)

    idx_step = 0 
    vis = Logger(cfg)
    for idx_epoch in range(cfg.start_epoch, cfg.n_epochs):
        idx_step = step(net, train_loader, optimizer, vis, idx_epoch, idx_step, cfg, 'train')
        step(net, val_loader, optimizer, vis, idx_epoch, idx_step, cfg, 'val')
        scheduler.step()
        save_dir = os.path.join(cfg.checkpoints_dir, cfg.exp_name, 'modelx' + str(cfg.scale_factor) + '.pth')
        torch.save({'epoch': idx_epoch + 1, 'state_dict': net.state_dict()}, save_dir)


def main(cfg):
    train_set = dataset.DataSetLoader(cfg)
    total_samples = len(train_set)
    train_indcs = range(total_samples)[int(0.1 * total_samples):]
    val_indcs = range(total_samples)[:int(0.1 * total_samples)]
    train_dataset = Subset(train_set, train_indcs)
    val_dataset = Subset(train_set, val_indcs)
    train_loader = DataLoader(dataset=train_dataset, num_workers=2, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, num_workers=2, batch_size=cfg.batch_size, shuffle=False)
    train_loader = DataLoader(dataset=train_set, num_workers=2, batch_size=cfg.batch_size, shuffle=True)
    train(train_loader, val_loader, cfg)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='Path of the config file')
    parser.add_argument('--data_dir', type=str, default='', help='Path of the dataset')
    parser.add_argument('--fast_test', default=False, action='store_true')
    args = parser.parse_args()
    cfg = cfg_parser(args)
    torch.manual_seed(0)
    np.random.seed(0)
    if cfg.fast_test:
        modify_opt_for_fast_test(cfg)
    main(cfg)

