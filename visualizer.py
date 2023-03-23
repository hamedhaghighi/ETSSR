import datetime
import os
import shutil
import sys
import time

from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


# def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
#     """Save images to the disk.

#     Parameters:
#         webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
#         visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
#         image_path (str)         -- the string is used to create image paths
#         aspect_ratio (float)     -- the aspect ratio of saved images
# width (int)              -- the images will be resized to width x width

#     This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
#     """
#     image_dir = webpage.get_image_dir()
#     short_path = ntpath.basename(image_path[0])
#     name = os.path.splitext(short_path)[0]

#     webpage.add_header(name)
#     ims, txts, links = [], [], []

#     for label, im_data in visuals.items():
#         im = util.tensor2im(im_data)
#         image_name = '%s_%s.png' % (name, label)
#         save_path = os.path.join(image_dir, image_name)
#         util.save_image(im, save_path, aspect_ratio=aspect_ratio)
#         ims.append(image_name)
#         txts.append(label)
#         links.append(image_name)
#     webpage.add_images(ims, txts, links, width=width)


class Logger():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, cfg):
        """Initialize the Visualizer class

        Parameters:
            cfg -- stores all the experiment flags; needs to be a subclass of Basecfgions
        Step 1: Cache the training/test cfgions
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        exp_path = os.path.join(cfg.checkpoints_dir, cfg.exp_name)
        if not cfg.load:
            if os.path.exists(exp_path):
                reply = ''

                while not reply.startswith('y') and not reply.startswith('n'):
                    reply = str(input(
                        f'exp_name {exp_path} exists. Do you want to delete it? (y/n): \n')).lower().strip()
                if reply.startswith('y'):
                    shutil.rmtree(exp_path)
                else:
                    exit(0)
        self.tb_dir = os.path.join(
            exp_path, 'TB', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(self.tb_dir, exist_ok=True)
        shutil.copy(cfg.cfg_path, exp_path)
        self.writer = SummaryWriter(self.tb_dir)
        self.log_name = os.path.join(
            cfg.checkpoints_dir, cfg.exp_name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write(
                '================ Training Loss (%s) ================\n' %
                now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def display_current_results(self, phase, visuals, g_step):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """

        for k, img in visuals.items():
            if k == 'real_A' and img.shape[1] == 6:
                rgb = img[:, 3:]
                fig = plt.figure()
                for i in range(min(2, rgb.shape[0])):
                    plt.subplot(1, 2, i + 1)
                    plt.imshow(
                        (rgb[i] *
                         0.5 +
                         0.5).permute(
                            1,
                            2,
                            0).cpu().detach().numpy())
                self.writer.add_figure(
                    phase + '/' + 'real_rgb', fig, g_step, True)
                img = img[:, :3]
            for j in range(img.shape[1]):
                fig = plt.figure()
                for i in range(min(2, img.shape[0])):
                    plt.subplot(1, 2, i + 1)
                    plt.imshow(
                        (img[i][j] * 0.5 + 0.5).cpu().detach().numpy(),
                        cmap='inferno' if k == 'range' else 'cividis',
                        vmin=0.0,
                        vmax=1.0)
                self.writer.add_figure(
                    phase + '/' + k + str(j), fig, g_step, True)

    def plot_current_losses(self, phase, epoch, losses, g_step):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        for tag, loss in losses.items():
            self.writer.add_scalar(phase + '/' + tag, loss, g_step)

        # plotting epoch
        self.writer.add_scalar('epoch', epoch, g_step)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, phase, epoch, iters, losses, tq=None):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = 'Validation\n' if phase is 'val' else ''
        message = message + '(epoch: %d, iters: %d) ' % (epoch, iters)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)
        if not tq:
            print(message)  # print the message
        else:
            tq.write(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
