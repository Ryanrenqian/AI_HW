import time

class TimerFactory:
    @staticmethod
    def produce(name):
        return zzy_timer(name)


class zzy_timer():
    def __init__(self, time):
        self.name = ''
        self.acc = 0
        self.t0 = 0
        self.last_record = 0
        self._hold_flag = False

    def start(self):
        if self._hold_flag:
            raise Exception('The timer has been holden. Please use resume().')
        self.acc = 0
        self.t0 = time.time()
        return self
    def hold(self):
        self.acc += time.time() - self.t0
        self._hold_flag = True
        return self

    def resume(self):
        self._hold_flag = False
        self.t0 = time.time()
        return self

    def end(self):
        self.acc += time.time() - self.t0
        self.last_record = self.acc
        self._hold_flag = False
        return self

    def get_last_record(self):
        return self.last_record

    def get_accumulated_time(self):
        return self.acc

    def __repr__(self):
        self.hold()
        acc_time = self.get_accumulated_time()
        self.resume()
        return f'{acc_time:.2f}'

    def __str__(self):
        self.hold()
        acc_time = self.get_accumulated_time()
        self.resume()
        return f'{self.name} : {acc_time:.2f}'

import numpy as np
import matplotlib.pyplot as plt
import os
import utils
# Import MNIST data




def plot_conv_feature_map(weights, name, channels_all=True, plot_dir ='./out/plots'):
    """
    Plots convolutional filters
    :param weights: numpy array of rank 4
    :param name: string, name of convolutional layer
    :param channels_all: boolean, optional
    :return: nothing, plots are saved on the disk
    """
    # make path to output folder
    #plot_dir = os.path.join(plot_dir, 'conv_weights')
    plot_dir = os.path.join(plot_dir, name)

    # create directory if does not exist, otherwise empty it
    utils.prepare_dir(plot_dir, empty=True)

    w_min = np.min(weights)
    w_max = np.max(weights)

    channels = [0]
    # make a list of channels if all are plotted
    if channels_all:
        channels = range(weights.shape[2])

    # get number of convolutional filters
    num_filters = weights.shape[3]

    # get number of grid rows and columns
    grid_r, grid_c = utils.get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

    # iterate channels

    # iterate channels
    for channel in channels:
        # iterate filters inside every channel
        for l, ax in enumerate(axes.flat):
            # get a single filter
            img = weights[:, :, channel, l]
            # put it on the grid
            ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap=plt.cm.gray)
            # remove any labels from the axes
            ax.set_xticks([])
            ax.set_yticks([])
        # save figure
        plt.savefig(os.path.join(plot_dir, '{}-{}.png'.format(name, channel)), bbox_inches='tight')

def plot_rgb_images(weights, name, channels_all=True, plot_dir ='./out/plots'):
    """
    Plots convolutional filters
    :param weights: numpy array of rank 4
    :param name: string, name of convolutional layer
    :param channels_all: boolean, optional
    :return: nothing, plots are saved on the disk
    """
    # make path to output folder
    #plot_dir = os.path.join(plot_dir, 'conv_weights')
    plot_dir = os.path.join(plot_dir, name)

    # create directory if does not exist, otherwise empty it
    utils.prepare_dir(plot_dir, empty=True)

    w_min = np.min(weights)
    w_max = np.max(weights)

    channels = [0]
    # make a list of channels if all are plotted
    if channels_all:
        channels = range(weights.shape[2])

    # get number of convolutional filters
    num_filters = weights.shape[3]

    # get number of grid rows and columns
    grid_r, grid_c = utils.get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))



    for l, ax in enumerate(axes.flat):
       # get a single filter
        img = weights[:, :, :, l]
        # put it on the grid
        ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap=plt.cm.gray)
        # remove any labels from the axes
        ax.set_xticks([])
        ax.set_yticks([])
    # save figure
    plt.savefig(os.path.join(plot_dir, '{}.png'.format(name)), bbox_inches='tight')

import torch
def denormalize(tensor, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010), inplace=False):
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device).reshape((1, 1, 3))
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device).reshape((1, 1, 3))
    tensor.mul_(std).add_(mean)
    return tensor

def denormalize1(tensor, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010), inplace=False):
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device).reshape((3, 1, 1))
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device).reshape((3, 1, 1))
    tensor.mul_(std).add_(mean)
    return tensor