# coding with UTF-8
# ******************************************
# *****CIFAR-10 with ResNet8 in Pytorch*****
# *****deconv_network.py               *****
# *****Author：Shiyi Liu               *****
# *****Time：  Oct 22nd, 2019          *****
# ******************************************
import torch
import torch.nn as nn


class OurConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, weight, kernel_size=3, stride=1, padding=1, output_padding=0, bias=False):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, output_padding=output_padding, bias=bias)
        self.weight.data = weight


class Deconv1(nn.Module):
    def __init__(self, inplanes, outplanes, conv_weight, padding=1, stride=1, groups=1, dilation=1):
        super().__init__()
        # self.weight = conv_weight
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = OurConvTranspose2d(in_channels=inplanes, out_channels=outplanes, weight=conv_weight,
                                          kernel_size=3, stride=stride, padding=padding, output_padding=0)
        # self.deconv1.weight.data = conv_weight

    def forward(self, x):
        out = self.relu(x)
        out = self.deconv1(out)
        return out


class Deconv_BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, deconv_weights, stride=1):
        super(Deconv_BasicBlock, self).__init__()
        self.deconv2 = OurConvTranspose2d(inplanes, inplanes, deconv_weights[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = OurConvTranspose2d(inplanes, outplanes, deconv_weights[1], kernel_size=3, stride=stride, padding=1, output_padding=stride-1, bias=False)
        self.shortcut = nn.Sequential()
        if inplanes != outplanes or stride != 1:
            self.shortcut = nn.Sequential(
                 OurConvTranspose2d(inplanes, outplanes, deconv_weights[2], kernel_size=1, stride=stride, output_padding=stride-1, padding=0, bias=False)
                 )

    def forward(self, x):
        x = self.relu(x)
        # print('\tbasic input: {}'.format(x.size()))
        out = self.relu(self.deconv2(x))
        # print('\tafter deconv2: {}'.format(out.size()))
        out = self.deconv1(out)
        # print('\tafter deconv1: {}'.format(out.size()))
        out += self.shortcut(x)
        return out


class Deconv_ResNet18(nn.Module):
    def __init__(self, deconv_weight, num_classes=10, deconv1=nn.Sequential()):
        self.inplanes = 512
        super(Deconv_ResNet18, self).__init__()
        self.deconv5_x = self._make_layer(256, deconv_weight[15:20], 2)
        self.deconv4_x = self._make_layer(128, deconv_weight[10:15], 2)
        self.deconv3_x = self._make_layer(64, deconv_weight[5:10], 2)
        self.deconv2_x = self._make_layer(64, deconv_weight[1:5], 1)
        self.deconv1 = deconv1

    def _make_layer(self, planes, basic_weights, stride):
        layers = []
        block_weight2 = [basic_weights[4], basic_weights[3]] if stride==2 else [basic_weights[3], basic_weights[2]]
        layers.append(Deconv_BasicBlock(self.inplanes, self.inplanes, block_weight2, stride=1))
        block_weight1 = [basic_weights[1], basic_weights[0], basic_weights[2]] if stride == 2 else [basic_weights[1],
                                                                                                    basic_weights[0]]
        layers.append(Deconv_BasicBlock(self.inplanes, planes, block_weight1, stride=stride))
        self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        # print('resnet input : {}'.format(x.size()))
        out = self.deconv5_x(x)
        # print('resnet after deconv5: {}'.format(out.size()))
        out = self.deconv4_x(out)
        # print('resnet after deconv4: {}'.format(out.size()))
        out = self.deconv3_x(out)
        # print('resnet after deconv3: {}'.format(out.size()))
        out = self.deconv2_x(out)
        # print('resnet after deconv2: {}'.format(out.size()))
        out = self.deconv1(out)

        return out
