import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from resnet import ResNet
import torch
import math
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
from torch import nn
from utils.guided_backprop import *
from resnet_sort_decon import plot_reconstruction
# Filter visualization
def get_grid_dim(x):
    """
    Transforms x into product of two integers
    :param x: int
    :return: two ints
    """
    factors = prime_powers(x)
    if len(factors) % 2 == 0:
        i = int(len(factors) / 2)
        return factors[i], factors[i - 1]

    i = len(factors) // 2
    return factors[i], factors[i]

def prime_powers(n):
    """
    Compute the factors of a positive integer
    Algorithm from https://rosettacode.org/wiki/Factors_of_an_integer#Python
    :param n: int
    :return: set
    """
    factors = set()
    for x in range(1, int(math.sqrt(n)) + 1):
        if n % x == 0:
            factors.add(int(x))
            factors.add(int(n // x))
    return sorted(factors)

def plot_conv_weights(weights,  channels_all=True):
    """
    Plots convolutional filters
    :param weights: numpy array of rank 4
    :param name: string, name of convolutional layer
    :param channels_all: boolean, optional
    :return: nothing, plots are saved on the disk
    """
    # create directory if does not exist, otherwise empty it

    w_min = np.min(weights)
    w_max = np.max(weights)

    channels = [0]
    # make a list of channels if all are plotted
    if channels_all:
        channels = range(weights.shape[1])

    # get number of convolutional filters
    num_filters = weights.shape[0]

    # get number of grid rows and columns
    grid_r, grid_c = get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

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

    return fig

def plot_conv_features(weights):
    """
    Plots convolutional features
    :param weights: numpy array of rank 4
    :param name: string, name of convolutional layer
    :return: nothing, plots are saved on the disk
    """
    # get number of convolutional filters
    num_filters = weights.shape[0]

    # get number of grid rows and columns
    grid_r, grid_c = get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

    for l, ax in enumerate(axes.flat):
        # get a single filter
        img = weights[l]
        # put it on the grid
        ax.imshow(img, interpolation='nearest', cmap='seismic')
        ax.set_xticks([])
        ax.set_yticks([])
    # plt.show()
    return fig


def filter_visualization(model,writer):
    # get the first layer
    layers = model.named_parameters()
    layer1 = next(layers)

    weight1 = layer1[1].detach().cpu().numpy()
    writer.add_figure('conv1 filter visualization',
                      plot_conv_weights(weight1))



def normalize(I):
    # 归一化梯度map，先归一化到 mean=0 std=1
    norm = (I - I.mean()) / I.std()
    # 把 std 重置为 0.1，让梯度map中的数值尽可能接近 0
    norm = norm * 0.1
    # 均值加 0.5，保证大部分的梯度值为正
    norm = norm + 0.5
    # 把 0，1 以外的梯度值分别设置为 0 和 1
    norm = norm.clip(0, 1)
    return norm

# functions to show an image
def imshow(img):
    fig=plt.figure()
    img = img / 2 + 0.5     # unnormalize
    npimg = normalize(img.numpy())

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    return fig

def resnet_conv(resnet, x):
    conv1 = nn.Sequential(*list(resnet.children())[0])(x).detach()
    conv5 = nn.Sequential(*list(resnet.children())[:-2])(x).detach()
    return conv1,conv5







if __name__ == '__main__':
    expid = 'cifar10_resnet18_experiment_1'
    epoch=16
    writer = SummaryWriter(log_dir='runs/%s' % expid)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'runs/%s/%d_epoch_para.pkl'%(expid,epoch)
    resnet = ResNet()
    resnet.load_state_dict(torch.load(model_path))
    filter_visualization(resnet, writer)
    #sample
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # load the image
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    imageloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                              shuffle=False,
                                              num_workers=2)
    dataiter = iter(imageloader)
    image, labels = dataiter.next()
    print(classes[labels[0]])
    imshow(torchvision.utils.make_grid(image[0]))

    writer.add_figure('original image',imshow(torchvision.utils.make_grid(image[0])))
    feature_conv1,feature_conv5 = resnet_conv(resnet,image)
    writer.add_figure('image in conv{}'.format(1),plot_conv_features(feature_conv1[0]))
    writer.add_figure('image in conv{}'.format(5),plot_conv_features(feature_conv5[0]))
    resnet=resnet.to(device)
    out1 = list(resnet.children())[0][0].cuda()(image.cuda())
    out2 = nn.Sequential(*list(resnet.children())[1:4])(out1)
    deconv_1, deconv_4 = plot_reconstruction(conv1_feature=out1, conv4_feature=out2, device=device)
    deconv_1 = deconv_1.squeeze( dim=0).cpu().detach()
    imshow(deconv_1)
    deconv_5 = deconv_4.squeeze(dim=0).cpu().detach()
    imshow(deconv_5)
    image=image[0].to(device)
    lr = 0.01
    # Descent 方法重建Conv1 只展示前16个filter结果
    # cnn_name = 'conv1'
    # for i in range(16):
    #     filter_pos = i
    #     layer_vis = Backprop(resnet, cnn_name, filter_pos, image, lr, cnn_name)
    #     layer_vis.visualise_layer_without_hooks()
    # # 重建Conv-5
    # cnn_name = 'layer4'
    # for i in range(16):
    #     filter_pos = i
    #     layer_vis = Backprop(resnet, cnn_name, filter_pos, image, lr, cnn_name)
    #     layer_vis.visualise_layer_without_hooks()

    # 利用反卷积重建
