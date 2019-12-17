#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:13:46 2019

@author: edsong
"""
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import os
import utils
from resnet import Resnet
import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pylab

PLOT_DIR = './out/plots'

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
    plot_dir = os.path.join(PLOT_DIR, 'conv_output')
    plot_dir = os.path.join(plot_dir, name)

    # create directory if does not exist, otherwise empty it
    utils.prepare_dir(plot_dir, empty=True)

    w_min = np.min(conv_img)
    w_max = np.max(conv_img)

    # get number of convolutional filters
    num_filters = conv_img.shape[0]

    # get number of grid rows and columns
    grid_r, grid_c = utils.get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

    # iterate filters
    for l, ax in enumerate(axes.flat):
        # get a single image
        img = conv_img[l, 0, :, :]
#        img = img.reshape((1,conv_img.shape[2],conv_img.shape[3]))
#        img = np.transpose(img,(1,2,0))
        # put it on the grid
        ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='bicubic', cmap='Greys')
#        ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='bicubic')

        # remove any labels from the axes
        ax.set_xticks([])
        ax.set_yticks([])
    # save figure
    plt.savefig(os.path.join(plot_dir, '{}.png'.format(name)), bbox_inches='tight')

def plot_image(conv_img, name):
    """
    Makes plots of results of performing convolution
    :param conv_img: numpy array of rank 4
    :param name: string, name of convolutional layer
    :return: nothing, plots are saved on the disk
    """
    # make path to output folder
    plot_dir = os.path.join(PLOT_DIR, 'conv_output')
    plot_dir = os.path.join(plot_dir, name)

    # create directory if does not exist, otherwise empty it
    utils.prepare_dir(plot_dir, empty=True)

    w_min = np.min(conv_img)
    w_max = np.max(conv_img)

    # get number of convolutional filters
    num_filters = conv_img.shape[0]

    # get number of grid rows and columns
    grid_r, grid_c = utils.get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

    # iterate filters
    for l, ax in enumerate(axes.flat):
        # get a single image
        img = conv_img[l, :, :, :]
        img = np.transpose(img,(1,2,0))
        # put it on the grid
#        ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='bicubic', cmap='Greys')
        ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='bicubic')

        # remove any labels from the axes
        ax.set_xticks([])
        ax.set_yticks([])
    # save figure
    plt.savefig(os.path.join(plot_dir, '{}.png'.format(name)), bbox_inches='tight')
    
def visualize_feature_maps(resnet,x):
    out1 = resnet.conv1(x)
    
    out = resnet.maxpool1(out1)
    out = resnet.block1(out)
    out = resnet.block2(out)
    out2 = out
    out = resnet.maxpool2(out)
    out = resnet.block3(out)
    out = resnet.block4(out)

    out = resnet.maxpool3(out)
    out = resnet.block5(out)
    out = resnet.block6(out)
    
    out = resnet.maxpool4(out)
    out = resnet.block7(out)
    out3 = resnet.block8(out)
    
    
    return out1,out2,out3


class block_backward(nn.Module):
    def __init__(self,inplane,outplane,kernel_size=3,stride=1,padding=1):
        super(block_backward, self).__init__()
        self.unconv2 = nn.ConvTranspose2d(inplane, inplane, 3, stride=stride, padding=padding,bias=False)
        self.unconv1 = nn.ConvTranspose2d(inplane, outplane, 3, stride=stride, padding=padding,bias=False) 
        return
    def forward(self,x):
    # ignoring shortcut
        x = F.relu(x)
        x = self.unconv2(x)
        x = F.relu(x)
        x = self.unconv1(x)
        return x

class shortcut_backward(nn.Module):
    def __init__(self,inplane,outplane,kernel_size=3,stride=1,padding=1):
        super(shortcut_backward, self).__init__()
        self.unconv1 = nn.ConvTranspose2d(inplane, outplane, 1, stride=stride,bias=False) 
        return
    def forward(self,x):
    # ignoring shortcut
        x = F.relu(x)
        x = self.unconv1(x)
        return x

class resnet_block2_backward(nn.Module):
    def __init__(self,param):
        super(resnet_block2_backward, self).__init__()
        self.param = param
        
        self.backblock2 = block_backward(64,64)
        self.backblock1 = block_backward(64,64)
        self.unpool1 = unpool()

        self.unconv1 = nn.ConvTranspose2d(64, 3, 7, stride=2, padding=3,bias=False)

        return
    def init(self):
        
        self.backblock2.unconv2.weight = self.param['module.block2.conv2.weight']
        self.backblock2.unconv1.weight = self.param['module.block2.conv1.weight']
        
        self.backblock1.unconv2.weight = self.param['module.block1.conv2.weight']
        self.backblock1.unconv1.weight = self.param['module.block1.conv1.weight']
        
        self.unconv1.weight = self.param['module.conv1.weight']
        
        return
        
    def forward(self,x):       
        
        x = self.backblock2(x)
        x = self.backblock1(x)
        x = self.unpool1(x)        

        x = self.unconv1(x)       
        return x
    
class resnet_block5_backward(nn.Module):
    def __init__(self,param):
        super(resnet_block5_backward, self).__init__()
        self.param = param
        self.backblock8 = block_backward(512,512)
        self.backblock7 = block_backward(512,256)
        self.unpool4 = unpool()
        
        
        self.backblock6 = block_backward(256,256)
        self.backblock5 = block_backward(256,128)
        self.unpool3 = unpool()
        
        self.backblock4 = block_backward(128,128)
        self.backblock3 = block_backward(128,64)
        self.unpool2 = unpool()
        
        self.backblock2 = block_backward(64,64)
        self.backblock1 = block_backward(64,64)
        self.unpool1 = unpool()

        self.unconv1 = nn.ConvTranspose2d(64, 3, 7, stride=2, padding=3,bias=False)

        return
    def init(self):
        self.backblock8.unconv2.weight = self.param['module.block8.conv2.weight']
        self.backblock8.unconv1.weight = self.param['module.block8.conv1.weight']
        
        self.backblock7.unconv2.weight = self.param['module.block7.conv2.weight']
        self.backblock7.unconv1.weight = self.param['module.block7.conv1.weight']
        
        self.backblock6.unconv2.weight = self.param['module.block6.conv2.weight']
        self.backblock6.unconv1.weight = self.param['module.block6.conv1.weight']
        
        self.backblock5.unconv2.weight = self.param['module.block5.conv2.weight']
        self.backblock5.unconv1.weight = self.param['module.block5.conv1.weight']
        
        self.backblock4.unconv2.weight = self.param['module.block4.conv2.weight']
        self.backblock4.unconv1.weight = self.param['module.block4.conv1.weight']
        
        self.backblock3.unconv2.weight = self.param['module.block3.conv2.weight']
        self.backblock3.unconv1.weight = self.param['module.block3.conv1.weight']
        
        self.backblock2.unconv2.weight = self.param['module.block2.conv2.weight']
        self.backblock2.unconv1.weight = self.param['module.block2.conv1.weight']
        
        self.backblock1.unconv2.weight = self.param['module.block1.conv2.weight']
        self.backblock1.unconv1.weight = self.param['module.block1.conv1.weight']
        
        self.unconv1.weight = self.param['module.conv1.weight']
        
        return
        
    def forward(self,x):
        x = self.backblock8(x)
        x = self.backblock7(x)
        x = self.unpool4(x)
        
        x = self.backblock6(x)
        x = self.backblock5(x)
        x = self.unpool3(x)        
        
        x = self.backblock4(x)
        x = self.backblock3(x)
        x = self.unpool2(x)        
        
        x = self.backblock2(x)
        x = self.backblock1(x)
        x = self.unpool1(x)        

        x = self.unconv1(x)       
        return x
    
class resnet_block5_shortcut_backward(nn.Module):
    def __init__(self,param):
        super(resnet_block5_shortcut_backward, self).__init__()
        self.param = param
        self.backblock8 = shortcut_backward(512,512)
        self.backblock7 = shortcut_backward(512,256)
        self.unpool4 = unpool()
        
        
        self.backblock6 = shortcut_backward(256,256)
        self.backblock5 = shortcut_backward(256,128)
        self.unpool3 = unpool()
        
        self.backblock4 = shortcut_backward(128,128)
        self.backblock3 = shortcut_backward(128,64)
        self.unpool2 = unpool()
        
        self.backblock2 = shortcut_backward(64,64)
        self.backblock1 = shortcut_backward(64,64)
        self.unpool1 = unpool()

        self.unconv1 = nn.ConvTranspose2d(64, 3, 7, stride=2, padding=3,bias=False)

        return
    def init(self):
        self.backblock8.unconv1.weight = self.param['module.block8.projection_shortcut.weight']
        
        self.backblock7.unconv1.weight = self.param['module.block7.projection_shortcut.weight']
        
        self.backblock6.unconv1.weight = self.param['module.block6.projection_shortcut.weight']
        
        self.backblock5.unconv1.weight = self.param['module.block5.projection_shortcut.weight']
        
        self.backblock4.unconv1.weight = self.param['module.block4.projection_shortcut.weight']
        
        self.backblock3.unconv1.weight = self.param['module.block3.projection_shortcut.weight']
        
        self.backblock2.unconv1.weight = self.param['module.block2.projection_shortcut.weight']
        
        self.backblock1.unconv1.weight = self.param['module.block1.projection_shortcut.weight']
        
        self.unconv1.weight = self.param['module.conv1.weight']
        
        return
        
    def forward(self,x):
        x = self.backblock8(x)
        x = self.backblock7(x)
        x = self.unpool4(x)
        
        x = self.backblock6(x)
        x = self.backblock5(x)
        x = self.unpool3(x)        
        
        x = self.backblock4(x)
        x = self.backblock3(x)
        x = self.unpool2(x)        
        
        x = self.backblock2(x)
        x = self.backblock1(x)
        x = self.unpool1(x)        

        x = self.unconv1(x)       
        return x
    
class unpool(nn.Module):
    def __init__(self):
        super(unpool, self).__init__()
        return
    def forward(self,x):
        sh = np.array(x.size()).tolist()
        dim = len(sh[2:])
        x = torch.transpose(x,1,2)
        x = torch.transpose(x,2,3)
        out = (torch.reshape(x, [-1] + [sh[2], sh[1]]))
        for i in range(dim, 0, -1):
            out = torch.cat((out, out),i)
        out_size = [-1] + [s * 2 for s in sh[2:]] + [sh[1]]
        out = torch.reshape(out, out_size)
        out = torch.transpose(out,2,3)
        out = torch.transpose(out,1,2)
        return out
    
def normalize(img):
    for i in range(img.shape[0]):
        minpixel = np.min(img[i,:,:,:])
        maxpixel = np.max(img[i,:,:,:])
        img[i,:,:,:] = ((img[i,:,:,:] - minpixel)*255/(maxpixel-minpixel))
        
    return img.astype('int32')


class InvertRepresentation():
    def __init__(self,pretrained_model):
        self.pretrain_model = pretrained_model
        for param in self.pretrain_model.parameters():
            param.requires_grad = False
        return
    def euclidian_loss(self, org_matrix, target_matrix):
        """
            Euclidian loss is the main loss function in the paper
            ||fi(x) - fi(x_0)||_2^2& / ||fi(x_0)||_2^2
        """
        distance_matrix = target_matrix - org_matrix
        euclidian_distance = self.alpha_norm(distance_matrix, 2)
        normalized_euclidian_distance = euclidian_distance / self.alpha_norm(org_matrix, 2)
        return normalized_euclidian_distance
    
    def alpha_norm(self, input_matrix, alpha):
        """
            Converts matrix to vector then calculates the alpha norm
        """
        alpha_norm = ((input_matrix.view(-1))**alpha).sum()
        return alpha_norm

    def total_variation_norm(self, input_matrix, beta):
        """
            Total variation norm is the second norm in the paper
            represented as R_V(x)
        """
        to_check = input_matrix[:, :-1, :-1]  # Trimmed: right - bottom
        one_bottom = input_matrix[:, 1:, :-1]  # Trimmed: top - right
        one_right = input_matrix[:, :-1, 1:]  # Trimmed: top - right
        total_variation = (((to_check - one_bottom)**2 +
                            (to_check - one_right)**2)**(beta/2)).sum()
        return total_variation
    
    def generate_inverted_image_specific_layer(self, featuremaps, img_size):
        opt_img = Variable(1e-1 * torch.randn(1, 3, img_size, img_size)).cuda()
        opt_img.requires_grad=True
        optimizer = optim.SGD([opt_img], lr=500, momentum=0.9)
        alpha_reg_alpha = 6
        # The multiplier, lambda alpha
        alpha_reg_lambda = 1e-7
    
        # Total variation regularization parameters
        # Parameter beta, which is actually second norm
        tv_reg_beta = 2
        # The multiplier, lambda beta
        tv_reg_lambda = 1e-8
        img = np.zeros((1,img_size, img_size,3))

        for i in range(1000):
            optimizer.zero_grad()
            # Get the output from the model after a forward pass until target_layer
            # with the generated image (randomly generated one, NOT the real image)
            out1,out2,out3 = visualize_feature_maps(self.pretrain_model, opt_img)
            # Calculate euclidian loss
            euc_loss = 1e-1 * self.euclidian_loss(torch.from_numpy(featuremaps).cuda(), out3)
            # Calculate alpha regularization
            reg_alpha = alpha_reg_lambda * self.alpha_norm(opt_img, alpha_reg_alpha)
#            Calculate total variation regularization
            reg_total_variation = tv_reg_lambda * self.total_variation_norm(opt_img,
                                                                            tv_reg_beta)
#             Sum all to optimize
            loss = euc_loss + reg_alpha + reg_total_variation
#            loss = euc_loss 
            
            # Step
            loss.backward()
            optimizer.step()
            # Generate image every 5 iterations
            if i % 50 == 0:
                print('Iteration:', str(i), 'Loss:', loss.data.cpu().numpy())
                img = opt_img.detach().cpu().numpy()
                img = normalize(img)
            #    img = np.reshape(img)
                img = np.transpose(img[0],(1,2,0))
                plt.imshow(img,interpolation='bicubic')
                plt.show()
#                recreated_im = recreate_image(opt_img)
#                im_path = '../generated/Inv_Image_Layer_' + str(target_layer) + \
#                    '_Iteration_' + str(i) + '.jpg'
#                save_image(recreated_im, im_path)
    
            # Reduce learning rate every 40 iterations
            if i % 5 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.97
#                    print(param_group['lr'])
        return opt_img
        
        
  
if __name__ == '__main__':

    transform = transforms.Compose([transforms.RandomResizedCrop(224),transforms.ToTensor() ,transforms.Normalize((0.5 , 0.5 , 0.5) , (0.5 , 0.5 , 0.5))])  
#    trainset = torchvision.datasets.CIFAR10(root = './data' , train = True , download = True , transform = transform)
#    trainloader = torch.utils.data.DataLoader(trainset , batch_size = 256 , shuffle = True , num_workers =2)    
    testset = torchvision.datasets.CIFAR10(root = './data' , train = False , download = True , transform = transform)
    testloader = torch.utils.data.DataLoader(testset , batch_size = 9 , shuffle = False , num_workers = 2)
#    classes = ('plane' , 'car' , 'bird' , 'cat' , 'deer' , 'dog' , 'frog' , 'horse' , 'ship' , 'truck')
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"
#    
#    
    resnet = Resnet()
    device_ids=[0]
    resnet = resnet.cuda(device_ids[0])
    net=torch.nn.DataParallel(resnet,device_ids=device_ids)
#
    # read model
    print('===> Try load checkpoint')
    if os.path.isdir('checkpoint'):
        try:
            checkpoint = torch.load('./checkpoint/resnet_final.t7')
            net.load_state_dict(checkpoint['state'])        
            start_epoch = checkpoint['epoch']
            print('===> Load last checkpoint data')
        except FileNotFoundError:
            start_epoch = 0
            print('Can\'t found resnet_final.t7')
    else:
        start_epoch = 0
        print('===> Start from scratch')
#    
    parm={}
    for name,parameters in net.named_parameters():
#        print(name,':',parameters.size())
        parm[name]=parameters
    
    
    conv_weights = parm['module.conv1.weight']
#    .detach().cpu().numpy()
#    plot_conv_weights(conv_weights, 'conv1.weight')
    
    unconv = nn.ConvTranspose2d(64, 3, 7, stride=2, padding=3,bias=False)
    unconv.weight = conv_weights
    
    back2_resnet = resnet_block2_backward(parm)
    back5_resnet = resnet_block5_backward(parm)
    back5_shortcut = resnet_block5_shortcut_backward(parm)
    back2_resnet.init()
    back5_resnet.init()
    back5_shortcut.init()
    
    
#
    # get output of all convolutional layers
    # here we need to provide an input image
    
    for data in testloader:
        images , labels = data
        out1,out2,out3 = visualize_feature_maps(resnet,Variable(images.cuda()))
        image = images.detach().cpu().numpy()
        image = normalize(image)
        plot_image(image, 'image')
        break
        #print(predicted.data[0])
    
#    conv1_out = out1.detach().cpu().numpy()
    conv2_out = out2.detach().cpu().numpy()

#    conv5_out = out3
    conv5_out = F.relu(out3).detach().cpu().numpy()
    conv1_out = F.relu(out1).detach().cpu().numpy()

    
    
    plot_conv_output(conv1_out, 'conv1_feature_maps')
    plot_conv_output(conv2_out, 'conv2_feature_maps')
    plot_conv_output(conv5_out, 'conv5_feature_maps')
    
#    plot_conv_output(conv_out, 'conv1')
    out1 = F.relu(out1)
    for i in range(3):
        isolated = out1.detach().cpu().numpy()
        isolated[:,:i,:,:] = 0
        isolated[:,i+1:,:,:] = 0   #除了channel i 以外全部变成0
#        print (np.shape(isolated))
#        totals = np.sum(isolated,axis=(1,2,3))
#        best = np.argmin(totals,axis=0)
#        print (best)
        # totals = np.sum(pixelactive,axis=(1,2,3))
        # best = np.argmax(totals,axis=0)
        # best = 0
        
        conv1_out_backward = F.relu(Variable(torch.from_numpy(isolated).cuda()))
        conv1_out_backward = unconv(conv1_out_backward)
        img_back = conv1_out_backward.detach().cpu().numpy()
        img_back = normalize(img_back)
        plot_image(img_back, 'conv1_backward_channel{}'.format(i))
        
    out2 = F.relu(out2)
    for i in range(3):
        isolated = out2.detach().cpu().numpy()
        isolated[:,:i,:,:] = 0
        isolated[:,i+1:,:,:] = 0   #除了channel i 以外全部变成0
        img_back_conv2 = back2_resnet(Variable(torch.from_numpy(isolated).cuda()))
        img_back_conv2 = img_back_conv2.detach().cpu().numpy()
        img_back_conv2 = normalize(img_back_conv2)
        plot_image(img_back_conv2, 'conv2_backward_channel{}'.format(i))
    
    
    out3 = F.relu(out3)
    # shortcut path backward
    for i in range(3):
        isolated = out3.detach().cpu().numpy()
        isolated[:,:i,:,:] = 0
        isolated[:,i+1:,:,:] = 0   #除了channel i 以外全部变成0
        img_back_conv5 = back5_resnet(Variable(torch.from_numpy(isolated).cuda()))
        img_back_conv5 = img_back_conv5.detach().cpu().numpy()
        img_back_conv5 = normalize(img_back_conv5)
        plot_image(img_back_conv5, 'conv5_residue_backward_channel{}'.format(i))
    
    # residue path backward
    for i in range(3):
        isolated = out3.detach().cpu().numpy()
        isolated[:,:i,:,:] = 0
        isolated[:,i+1:,:,:] = 0   #除了channel i 以外全部变成0
        img_back_conv5 = back5_shortcut(Variable(torch.from_numpy(isolated).cuda()))
        img_back_conv5 = img_back_conv5.detach().cpu().numpy()
        img_back_conv5 = normalize(img_back_conv5)
        plot_image(img_back_conv5, 'conv5_shortcut_backward_channel{}'.format(i))
        
    
    
    
    # back training visualization
#    inverted_representation = InvertRepresentation(resnet)
#    img = inverted_representation.generate_inverted_image_specific_layer(conv5_out[0,:,:,:],224)
#    img = img.detach().cpu().numpy()
#    img = normalize(img)
##    img = np.reshape(img)
#    img = np.transpose(img[0],(1,2,0))
#    plt.imshow(img,interpolation='bicubic')
#    plt.show()
    
        