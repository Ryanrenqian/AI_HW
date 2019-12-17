import torch
import torch.nn as nn

import numpy as np
import argparse

class BasicBlock(nn.Module):
    # For Resnet18 or34
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1):
        super(BasicBlock, self).__init__()
        # Main branch of the BasicBlock
        self.basic = nn.Sequential(
             nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
             nn.BatchNorm2d(out_channel),
             nn.ReLU(inplace=True),
             nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1 , padding=1, bias=False),
             nn.BatchNorm2d(out_channel))

        # shortcut of the BasicBlock
        self.short_cut = nn.Sequential()
        # keep the size and channel of shortcut consistent
        if in_channel != self.expansion * out_channel or stride != 1:
            self.short_cut = nn.Sequential(
                 nn.Conv2d(in_channel, self.expansion * out_channel, kernel_size=1, stride=stride, bias=False),
                 nn.BatchNorm2d(self.expansion * out_channel))

    def forward(self, x):
        output = self.basic(x)
        output += self.short_cut(x)
        # keep the input a tensor
        output = torch.nn.functional.relu(output)
        return output


class Resnet18(nn.Module):
    def __init__(self, Block,  num_blocks, num_classes=10):
        super(Resnet18, self).__init__()
        self.in_channel = 64
        self.Conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.Maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        self.Conv2_x = self._make_layer(Block, 64, num_blocks[0], 1)
        self.Conv3_x = self._make_layer(Block, 128, num_blocks[1], 2)
        self.Conv4_x = self._make_layer(Block, 256, num_blocks[2], 2)
        self.Conv5_x = self._make_layer(Block, 512, num_blocks[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Block.expansion, num_classes)


    def _make_layer(self, Block, out_channel, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(Block(self.in_channel, out_channel, stride))
            self.in_channel = Block.expansion * out_channel
        return nn.Sequential(*layers)


    def forward(self, x):
        output = self.Conv1(x)
        output, indices = self.Maxpool(output)
        output = self.Conv2_x(output)
        output = self.Conv3_x(output)
        output = self.Conv4_x(output)
        output = self.Conv5_x(output)
        output = self.avg_pool(output)
        #Change the shape of the sensor, [batchsize,512]
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

    def visual_conv1(self, x):
        visual_output = self.Conv1(x)
        return visual_output

    def visual_conv5(self, x):
        visual_output = self.Conv1(x)
        visual_output, indices = self.Maxpool(visual_output)
        visual_output = self.Conv2_x(visual_output)
        visual_output = self.Conv3_x(visual_output)
        visual_output = self.Conv4_x(visual_output)
        visual_output = self.Conv5_x(visual_output)
        return visual_output, indices
def Resnet_18():

    return Resnet18(BasicBlock, [2, 2, 2, 2])
