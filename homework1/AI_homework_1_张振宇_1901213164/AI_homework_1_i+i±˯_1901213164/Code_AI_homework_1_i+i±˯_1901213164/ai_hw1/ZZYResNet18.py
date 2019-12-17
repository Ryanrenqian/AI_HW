import torch
import torch.nn as nn
import functools

import torch
import torchvision
import torchvision.transforms as transforms


conv3x3 = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
conv1x1 = functools.partial(nn.Conv2d, kernel_size=1, padding=0)
bn = nn.BatchNorm2d


class BasicBlock(nn.Module):
    def __init__(self, in_chanels, out_chanels):
        super().__init__()
        self.down_flag = (in_chanels * 2 == out_chanels)
        if self.down_flag == False and in_chanels != out_chanels: raise Exception('please check in_chanels and out_chanels')
        self.conv1 = conv3x3(in_chanels, out_chanels, stride=2 if self.down_flag else 1)
        self.conv2 = conv3x3(out_chanels, out_chanels, stride=1)
        self.bn1 = bn(out_chanels)
        self.bn2 = bn(out_chanels)
        if self.down_flag:
            self.down_conv = conv1x1(in_chanels, out_chanels, stride=2)
            self.down_bn = bn(out_chanels)
        self.relu = nn.ReLU()
        pass
    def forward(self, input):
        x = input
        res = self.conv1(input)
        res = self.bn1(res)
        res = self.relu(res)
        res = self.conv2(res)
        res = self.bn2(res)
        if self.down_flag:
            x = self.down_conv(x)
            x = self.down_bn(x)
        output = res + x
        output = self.relu(output)
        return output

def make_conv_module(in_chanels, out_chanels):
    bb1 = BasicBlock(in_chanels, out_chanels)
    bb2 = BasicBlock(out_chanels, out_chanels)
    return nn.Sequential(bb1, bb2)

class ZZYResNet18(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = bn(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.conv2_x = make_conv_module(64, 64)
        self.conv3_x = make_conv_module(64, 128)
        self.conv4_x = make_conv_module(128, 256)
        self.conv5_x = make_conv_module(256, 512)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, n_classes)
    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.maxpool(output)

        output = self.conv2_x(output)

        output = self.conv3_x(output)

        output = self.conv4_x(output)

        output = self.conv5_x(output)

        output = self.avgpool(output)
        output = torch.flatten(output, 1)
        output = self.fc(output)
        return output

class ZZYResNet18_indices(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = bn(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1, return_indices = True)

        self.conv2_x = make_conv_module(64, 64)
        self.conv3_x = make_conv_module(64, 128)
        self.conv4_x = make_conv_module(128, 256)
        self.conv5_x = make_conv_module(256, 512)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, n_classes)
    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu(output)
        output, _ = self.maxpool(output)

        output = self.conv2_x(output)

        output = self.conv3_x(output)

        output = self.conv4_x(output)

        output = self.conv5_x(output)

        output = self.avgpool(output)
        output = torch.flatten(output, 1)
        output = self.fc(output)
        return output