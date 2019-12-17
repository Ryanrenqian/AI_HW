'''
zhuliwen: liwenzhu@pku.edu.cn
October 24，2019
ref: https://blog.csdn.net/jacke121/article/details/85422244
     https://github.com/kvfrans/feature-visualization
'''

import cv2
import numpy as np
import torch
from torch.autograd import Variable
from AI_homework_1 import ResNet, BasicBlock
import os
import matplotlib.pyplot as plt
import torch.nn as nn


def preprocess_image(cv2im, resize_im=True):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        cv2im = cv2.resize(cv2im, (224, 224))
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var

class FeatureVisualization():
    def __init__(self, img_path, selected_layer, model, chan=0):
        self.img_path = img_path
        self.selected_layer = selected_layer
        self.pretrained_model = model.named_modules()
        self.chan = chan

    def process_image(self):
        img = cv2.imread(self.img_path)
        img = preprocess_image(img)
        return img

    def get_feature(self):
        # input = Variable(torch.randn(1, 3, 224, 224))
        input = self.process_image()
        print(input.shape)
        x = input.cuda()

        if self.selected_layer == 'conv1':
            for index, layer in enumerate(self.pretrained_model):
                if layer[0] in ['conv1.0']:
                    x = layer[1](x)
                    return x

        elif self.selected_layer == 'conv5-x':
            for index, layer in enumerate(self.pretrained_model):
                if layer[0] in ['conv1', 'conv2_x', 'conv3_x', 'conv4_x', 'conv5_x.0']:
                    x = layer[1](x)
                    if (layer[0] == 'conv5_x.0' ):
                        return x


    def get_single_feature(self):
        features = self.get_feature()
        print(features.shape)

        feature = features[:, self.chan, :, :] # 在这里可以提取 64 个
        # feature = features[self.chan, :, :]

        feature = feature.view(feature.shape[1], feature.shape[2])
        print(feature.shape)

        return feature

    def get_deconv_feature(self):
        feature = self.get_feature()
        if self.selected_layer == 'conv1':
            unConv1 = nn.ConvTranspose2d(64, 3, kernel_size=7, padding=3, stride=2, bias=False).cuda()
            deconv_features = unConv1(feature)
            deconv_feature = deconv_features[:, 0, :, :]
            deconv_feature = deconv_feature.view(deconv_feature.shape[1], deconv_feature.shape[2])
            return deconv_feature
        elif self.selected_layer == 'conv5-x':
            unConv_5 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding = 1,bias=False).cuda()
            unBN_5 = nn.BatchNorm2d(256).cuda()
            unReLU_5 = nn.ReLU(inplace=True).cuda()
            unConv_4 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding = 1, bias=False).cuda()
            unBN_4 = nn.BatchNorm2d(128).cuda()
            unReLU_4 = nn.ReLU(inplace=True).cuda()
            unConv_3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding = 1, bias=False).cuda()
            unBN_3 = nn.BatchNorm2d(64).cuda()
            unReLU_3 = nn.ReLU(inplace=True).cuda()
            unConv_2 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding = 1, bias=False).cuda()
            unBN_2 = nn.BatchNorm2d(64).cuda()
            unReLU_2 = nn.ReLU(inplace=True).cuda()
            unConv1 = nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3, bias=False).cuda()

            deconv_features = unConv_5(feature)
            deconv_features = unBN_5(deconv_features)
            deconv_features = unReLU_5(deconv_features)
            deconv_features = unConv_4(deconv_features)
            deconv_features = unBN_4(deconv_features)
            deconv_features = unReLU_4(deconv_features)
            deconv_features = unConv_3(deconv_features)
            deconv_features = unBN_3(deconv_features)
            deconv_features = unReLU_3(deconv_features)
            deconv_features = unConv_2(deconv_features)
            deconv_features = unBN_2(deconv_features)
            deconv_features = unReLU_2(deconv_features)

            deconv_features = unConv1(deconv_features)
            deconv_feature = deconv_features[:, 0, :, :]
            deconv_feature = deconv_feature.view(deconv_feature.shape[1], deconv_feature.shape[2])
            return deconv_feature

    def save_feature_to_img(self, file, ax1):
        feature = self.get_deconv_feature()
        feature = feature.data.cpu().detach().numpy()

        # use sigmod to [0,1]
        feature = 1.0 / (1 + np.exp(-1 * feature))

        # to [0,255]
        feature = np.round(feature * 255)
        print(os.path.splitext(file))

        w_min = np.min(feature)
        w_max = np.max(feature)

        print(feature)
        if self.selected_layer == 'conv1':
            feature = cv2.applyColorMap(np.uint8(feature), cv2.COLORMAP_JET)

        ax1.imshow(feature, vmin=w_min, vmax=w_max,
                       interpolation='nearest')  # 'gist_ncar'

        ax1.set_xticks([])
        ax1.set_yticks([])


if __name__ == '__main__':
    net = ResNet(BasicBlock, [2, 2, 2, 2])
    net = net.cuda()
    net.load_state_dict(torch.load('resnet18-25-best.pth'))


    for file in os.listdir('./img_fmap'):
        if 'after' not in file:
            # for layer in ['conv1', 'conv5-x']:
            for layer in ['conv1']:
                fig, axes = plt.subplots(8, 8, figsize=(10, 10))
                for i, ax1 in enumerate(axes.flat):
                    myClass1 = FeatureVisualization('./img_fmap/'+ file, layer, net, chan=i) # conv1 or conv5-x
                    print(myClass1.pretrained_model)
                    myClass1.save_feature_to_img('./img_fmap/'+ file, ax1)
                plt.show()
                fig.savefig('{}_{}_after_Deconv1.jpg'.format('./img_fmap/'+ os.path.splitext(file)[0], layer), bbox_inches = 'tight')

    for file in os.listdir('./img_fmap'):
        if 'after' not in file:
            for layer in ['conv5-x']:
                fig, ax1 = plt.subplots()
                myClass1 = FeatureVisualization('./img_fmap/'+ file, layer, net, chan=0) # conv1 or conv5-x
                print(myClass1.pretrained_model)
                myClass1.save_feature_to_img('./img_fmap/'+ file, ax1)
                fig.savefig('{}_{}_after_Deconv5.jpg'.format('./img_fmap/' + os.path.splitext(file)[0], layer),
                            bbox_inches='tight')