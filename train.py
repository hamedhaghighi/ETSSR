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

def train(train_loader, cfg):
    IC = cfg.input_channel
    net = Net(cfg.scale_factor, IC).to(cfg.device)
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

    loss_list = []
    idx_step = 0 
    loss_list = defaultdict(list)
    vis = Logger(cfg)
    for idx_epoch in range(cfg.start_epoch, cfg.n_epochs):
        train_dl = iter(train_loader)
        n_trainbatch = len(train_loader) if not cfg.fast_test else 2
        train_tq = tqdm.tqdm(total=n_trainbatch, desc='Iter', position=3)
        for _ in range(len(train_tq)):
            HR_left, HR_right, LR_left, LR_right = next(train_dl)
            HR_left, HR_right, LR_left, LR_right = HR_left.to(cfg.device), HR_right.to(cfg.device),LR_left.to(cfg.device), LR_right.to(cfg.device)
            loss = net.calc_loss(LR_left, LR_right, HR_left, HR_right, cfg)
            for k ,v in net.get_losses().items():
                loss_list[k].append(v)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx_step % 50 == 0:
                avg_loss_list = {k: np.array(v).mean() for k , v in loss_list.items()}
                vis.print_current_losses('train', idx_epoch, idx_step, avg_loss_list, train_tq)
                vis.plot_current_losses('train', idx_epoch, avg_loss_list, idx_step)
                loss_list = defaultdict(list)

            train_tq.update(1)
            idx_step = idx_step + 1

        scheduler.step()
        save_dir = os.path.join(cfg.checkpoints_dir, cfg.exp_name, 'modelx' + str(cfg.scale_factor) + '.pth')
        torch.save({'epoch': idx_epoch + 1, 'state_dict': net.state_dict()}, save_dir)


def main(cfg):
    train_set = dataset.DataSetLoader(cfg)
    train_loader = DataLoader(dataset=train_set, num_workers=2, batch_size=cfg.batch_size, shuffle=True)
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

