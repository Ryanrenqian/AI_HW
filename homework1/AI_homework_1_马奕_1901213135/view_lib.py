import numpy as np
import matplotlib.pyplot as plt
import os
import utils
import torch

PLOT_DIR = './out_visualization'


def plot_conv_weights(weights, name, channels_all=True):
    """
    Plots convolutional filters
    :param weights: numpy array of rank 4
    :param name: string, name of convolutional layer
    :param channels_all: boolean, optional
    :return: nothing, plots are saved on the disk
    """
    # make path to output folder
    plot_dir = os.path.join(PLOT_DIR, 'conv_weights')
    plot_dir = os.path.join(plot_dir, name)

    # create directory if does not exist, otherwise empty it
    utils.prepare_dir(plot_dir, empty=True)

    w_min = np.min(weights)
    w_max = np.max(weights)

    channels = [0]
    # make a list of channels if all are plotted
    if channels_all:
        channels = range(weights.shape[1])

    # get number of convolutional filters
    num_filters = weights.shape[0]

    # get number of grid rows and columns
    grid_r, grid_c = utils.get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]), figsize=(12, 8))

    # iterate channels
    for channel in channels:
        # iterate filters inside every channel
        for l, ax in enumerate(axes.flat):
            # get a single filter
            img = weights[l, channel, :, :]
            # put it on the grid
            ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
            # remove any labels from the axes
            ax.set_xticks([])
            ax.set_yticks([])
        # save figure
        plt.savefig(os.path.join(plot_dir, '{}-{}.png'.format(name, channel)), bbox_inches='tight')


def plot_conv_output(conv_img, name):
    """
    Makes plots of results of performing convolution
    :param conv_img: numpy array of rank 4
    :param name: string, name of convolutional layer
    :return: nothing, plots are saved on the disk
    """
    # make path to output folder
    plot_dir = os.path.join(PLOT_DIR, 'feature_maps')
    plot_dir = os.path.join(plot_dir, name)
    plt.figure(num=None, figsize=(16, 12), dpi=80)
    # create directory if does not exist, otherwise empty it
    utils.prepare_dir(plot_dir, empty=True)

#    w_min = np.min(conv_img)
#    w_max = np.max(conv_img)

    # get number of convolutional filters
    num_filters = conv_img.shape[1]

    # get number of grid rows and columns
    grid_r, grid_c = utils.get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]), figsize=(12, 8))

    # iterate filters
    for l, ax in enumerate(axes.flat):
        # get a single image
        img = conv_img[0, l, :, :]
        # put it on the grid
        ax.imshow(img,  interpolation='bicubic', cmap='Greys')
        # remove any labels from the axes
        ax.set_xticks([])
        ax.set_yticks([])
    # save figure
    plt.savefig(os.path.join(plot_dir, '{}.png'.format(name)), bbox_inches='tight')



