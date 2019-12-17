'''
zhuliwen: liwenzhu@pku.edu.cn
October 24，2019
ref: https://github.com/grishasergei/conviz
'''

import matplotlib.pyplot as plt
import math

import torch
import numpy as np
from AI_homework_1 import ResNet, BasicBlock


def plot_conv_weights(weights, input_channel=0):
    w = weights
    w_min = np.min(w)
    w_max = np.max(w)

    num_filters = w.shape[3]
    num_grids = math.ceil(math.sqrt(num_filters))

    fig, axes = plt.subplots(num_grids, num_grids, figsize=(10, 10))

    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            img = w[:, :, input_channel, i]
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')

        ax.set_xticks([])
        ax.set_yticks([])


    plt.show()
    fig.savefig('./img_filter/filter_channel_{}.png'.format(input_channel), bbox_inches = 'tight')

if __name__ == '__main__':
    net = ResNet(BasicBlock, [2, 2, 2, 2])
    net.load_state_dict(torch.load('resnet18-158-best.pth'))

    parm = {}
    for name, parameters in net.named_parameters():
        parm[name] = parameters.detach().numpy()

    weights_conv1 = parm['conv1.0.weight'].transpose(2, 3, 1, 0)

    print('shape:', weights_conv1.shape)
    print('weights', weights_conv1)

    for channel in range(3):
        plot_conv_weights(weights=weights_conv1, input_channel = channel)