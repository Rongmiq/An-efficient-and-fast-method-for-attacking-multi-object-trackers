"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""

import time
from torch.utils.data import DataLoader
import os
import cv2
import numpy as np
import torch
import argparse
from options.train_options import TrainOptions
from CenterNet.src.lib.opts import opts
from util.visualizer import Visualizer
from models import create_model
from data_utils import PrefetchDataset
# from CenterNet.src.lib.test import PrefetchDataset as TestDataset
from CenterNet.src.lib.datasets.dataset.coco import COCO
from CenterNet.src.lib.detectors.ctdet import CtdetDetector

def print_current_losses_(epoch, iters, losses, t_comp, t_data):
    """print current losses on console; also save the losses to the disk

    Parameters:
        epoch (int) -- current epoch
        iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
        losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
        t_comp (float) -- computational time per data point (normalized by batch_size)
        t_data (float) -- data loading time per data point (normalized by batch_size)
    """
    message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
    for k, v in losses.items():
        message += '%s: %.3f ' % (k, v)

    print(message)  # print the message


if __name__ == '__main__':
    opt_train = TrainOptions().parse()
    opt_tracker = opts().parse()  # get training options
    print(opt_train,opt_tracker)
    ''' create and initialize model'''
    model = create_model(opt_train)      # create a model given opt.model and other options
    model.setup(opt_train)               # regular setup: load and print networks; create schedulers
    ''' create dataset and detector'''
    dataset = COCO(opt_tracker, split='train_half')
    opt = opts().update_dataset_info_and_set_heads(opt_tracker, dataset)


    detector = CtdetDetector(opt)
    print('Setting up train data...')
    train_loader = torch.utils.data.DataLoader(
        PrefetchDataset(opt_tracker, dataset, pre_process_func=None),batch_size=opt_train.batch_size, shuffle=True, num_workers=opt_train.num_workers, pin_memory=True)
    dataset_size = len(dataset)
    '''visualizer'''
    visualizer = Visualizer(opt_train)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    print('Starting training...')
    for epoch in range(opt_train.epoch_count, opt_train.epoch_count + opt_train.epochs):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for ind, pre_processed_images in enumerate(train_loader):  # inner loop within one epoch
            image = pre_processed_images.permute(0,3,1,2).cuda() #[b,h,w,c]
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt_train.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt_train.batch_size
            epoch_iter += opt_train.batch_size
            '''ATTENTION'''
            model.set_input(image)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt_train.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt_train.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt_train.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt_train.batch_size

                print_current_losses_(epoch, epoch_iter, losses, t_comp, t_data)
                if opt_train.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt_train.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt_train.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        '''change fooling loss's weight at the end of each epoch'''
        # model.update_weight()
        # print('weight in epoch %d is %f'%(epoch,model.weight))
        if epoch % opt_train.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt_train.niter + opt_train.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
