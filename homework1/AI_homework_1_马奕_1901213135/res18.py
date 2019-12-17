import numpy as np
import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict

scut = {}


class ResBlock(nn.Module):

    def __init__(self, inchannels, outchannels, stride=2, idx=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannels)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(outchannels)
        )

        #        scut[idx]=0
        self.idx = idx
        # print(idx)

    def forward(self, x):
        scut[self.idx] = self.shortcut(x)
        out = self.block(x)
        out += scut[self.idx]
        out = nn.functional.relu(out)
        return out


# class ResBlock(nn.Module):
#
#    def __init__(self, inchannels, outchannels, stride=2, idx=0):
#        super().__init__()
#        self.block = nn.Sequential(
#            nn.Conv2d(inchannels, outchannels, kernel_size=3, stride=stride, padding=1, bias=False),
#            nn.BatchNorm2d(outchannels),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=1, bias=False),
#            nn.BatchNorm2d(outchannels)
#        )
#        self.shortcut = nn.Sequential(
#            nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=stride, bias=False),
#            nn.BatchNorm2d(outchannels)
#        )
#
#        scut[idx]=0
#        self.idx=idx
#
#    def forward(self, x):
#        scut[self.idx]=self.shortcut(x)
#        out = self.block(x)
#        out += scut[self.idx]
#        out = nn.functional.relu(out)
#        return out
#
#


class Firstlayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.first(x)

        return out


class Lastlayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        out = self.pool(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def reslayer(inchannels, outchannels, stride, idx):
    blk = [ResBlock(inchannels, outchannels, stride=stride, idx=idx),
           ResBlock(outchannels, outchannels, stride=1, idx=idx + 1)]
    return nn.Sequential(*blk)


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            Firstlayer(),
            reslayer(64, 64, stride=2, idx=0),
            reslayer(64, 128, stride=2, idx=2),
            reslayer(128, 256, stride=2, idx=4),
            reslayer(256, 512, stride=2, idx=6),
        )
        # index of conv
        self.classifier = Lastlayer()
        # feature maps
        self.feature_maps = OrderedDict()
        # switch

    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)
        return out
