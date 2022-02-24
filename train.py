from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import argparse
from utils import *
from model import *
import dataset
from visualizer import Logger
from collections import defaultdict
import tqdm
import yaml
import os

def modify_opt_for_fast_test(opt):
    opt.n_epochs = 2
    opt.batch_size = 2
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

def train(train_loader, cfg):
    input_channel = 10 if cfg.train_on_sim else 3
    net = Net(cfg.scale_factor, input_channel).to(cfg.device)
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
    criterion_L1 = torch.nn.L1Loss().to(cfg.device)
    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)

    loss_epoch = []
    loss_list = []
    idx_step = 0 
    loss_dict = defaultdict(list)
    vis = Logger(cfg)
    for idx_epoch in range(cfg.start_epoch, cfg.n_epochs):
        train_dl = iter(train_loader)
        n_trainbatch = len(train_loader) if not cfg.fast_test else 2
        train_tq = tqdm.tqdm(total=n_trainbatch, desc='Iter', position=3)
        for _ in range(len(train_tq)):
            HR_left, HR_right, LR_left, LR_right = next(train_dl)
            b, c, h, w = LR_left[:, :3].shape
            HR_left, HR_right, LR_left, LR_right = HR_left.to(cfg.device)[:, :3], HR_right.to(cfg.device)[:, :3],\
                LR_left.to(cfg.device), LR_right.to(cfg.device)
            SR_left, SR_right, (M_right_to_left, M_left_to_right), (V_left, V_right) = net(LR_left, LR_right, is_training=1)

            ''' SR Loss '''
            loss_SR = criterion_L1(SR_left, HR_left) + criterion_L1(SR_right, HR_right)
            loss_dict['SR'].append(loss_SR.data.cpu())
            ''' Photometric Loss '''
            Res_left = torch.abs(HR_left - F.interpolate(LR_left[:, :3], scale_factor=scale, mode='bicubic', align_corners=False))
            Res_left = F.interpolate(Res_left, scale_factor=1 / scale, mode='bicubic', align_corners=False)
            Res_right = torch.abs(HR_right - F.interpolate(LR_right[:, :3], scale_factor=scale, mode='bicubic', align_corners=False))
            Res_right = F.interpolate(Res_right, scale_factor=1 / scale, mode='bicubic', align_corners=False)
            Res_leftT = torch.bmm(M_right_to_left.contiguous().view(b * h, w, w), Res_right.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                  ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            Res_rightT = torch.bmm(M_left_to_right.contiguous().view(b * h, w, w), Res_left.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                   ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            loss_photo = criterion_L1(Res_left * V_left.repeat(1, 3, 1, 1), Res_leftT * V_left.repeat(1, 3, 1, 1)) + \
                         criterion_L1(Res_right * V_right.repeat(1, 3, 1, 1), Res_rightT * V_right.repeat(1, 3, 1, 1))
            loss_dict['photo'].append(loss_photo.data.cpu())
            ''' Smoothness Loss '''
            loss_h = criterion_L1(M_right_to_left[:, :-1, :, :], M_right_to_left[:, 1:, :, :]) + \
                     criterion_L1(M_left_to_right[:, :-1, :, :], M_left_to_right[:, 1:, :, :])
            loss_w = criterion_L1(M_right_to_left[:, :, :-1, :-1], M_right_to_left[:, :, 1:, 1:]) + \
                     criterion_L1(M_left_to_right[:, :, :-1, :-1], M_left_to_right[:, :, 1:, 1:])
            loss_smooth = loss_w + loss_h
            loss_dict['smooth'].append(loss_smooth.data.cpu())
            ''' Cycle Loss '''
            Res_left_cycle = torch.bmm(M_right_to_left.contiguous().view(b * h, w, w), Res_rightT.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                       ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            Res_right_cycle = torch.bmm(M_left_to_right.contiguous().view(b * h, w, w), Res_leftT.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                        ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            loss_cycle = criterion_L1(Res_left * V_left.repeat(1, 3, 1, 1), Res_left_cycle * V_left.repeat(1, 3, 1, 1)) + \
                         criterion_L1(Res_right * V_right.repeat(1, 3, 1, 1), Res_right_cycle * V_right.repeat(1, 3, 1, 1))
            loss_dict['cycle'].append(loss_cycle.data.cpu())

            ''' Consistency Loss '''
            SR_left_res = F.interpolate(torch.abs(HR_left - SR_left), scale_factor=1 / scale, mode='bicubic', align_corners=False)
            SR_right_res = F.interpolate(torch.abs(HR_right - SR_right), scale_factor=1 / scale, mode='bicubic', align_corners=False)
            SR_left_resT = torch.bmm(M_right_to_left.detach().contiguous().view(b * h, w, w), SR_right_res.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                     ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            SR_right_resT = torch.bmm(M_left_to_right.detach().contiguous().view(b * h, w, w), SR_left_res.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                      ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            loss_cons = criterion_L1(SR_left_res * V_left.repeat(1, 3, 1, 1), SR_left_resT * V_left.repeat(1, 3, 1, 1)) + \
                       criterion_L1(SR_right_res * V_right.repeat(1, 3, 1, 1), SR_right_resT * V_right.repeat(1, 3, 1, 1))
            loss_dict['cons'].append(loss_cons.data.cpu())
            ''' Total Loss '''
            loss = loss_SR + 0.1 * loss_cons + 0.1 * (loss_photo + loss_smooth + loss_cycle)
            loss_dict['total_loss'].append(loss.data.cpu())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_tq.update(1)
            idx_step = idx_step + 1

            if idx_step % 50 == 0:
                # vis.plot_current_losses('train', idx_epoch, , idx_step)
                avg_loss_dict = {k: np.array(v).mean() for k , v in loss_dict.items()}
                vis.print_current_losses('train', idx_epoch, idx_step, avg_loss_dict, train_tq)
                vis.plot_current_losses('train', idx_epoch, avg_loss_dict, idx_step)
                loss_dict = defaultdict(list)
            loss_epoch.append(loss.data.cpu())

        scheduler.step()
        save_dir = os.path.join(cfg.checkpoints_dir, cfg.exp_name, 'modelx' + str(cfg.scale_factor) + '.pth')
        torch.save({'epoch': idx_epoch + 1, 'state_dict': net.state_dict()}, save_dir)


def main(cfg):
    train_set = dataset.DataSetLoader(cfg)
    train_loader = DataLoader(dataset=train_set, num_workers=6, batch_size=cfg.batch_size, shuffle=True)
    train(train_loader, cfg)

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

