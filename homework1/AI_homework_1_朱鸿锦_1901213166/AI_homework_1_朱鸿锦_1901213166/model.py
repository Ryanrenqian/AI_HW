# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class BasicBlock(nn.Module):
    
    def __init__(self,in_channels,out_channels,stride):
        super().__init__()
        
        self.residual_function = nn.Sequential(
                #nn.MaxPool2d(kernel_size=3,stride=2),
                nn.Conv2d(in_channels,out_channels,kernel_size=3,padding = 1,stride = stride,bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels,out_channels,kernel_size=3,padding = 1,stride = 1,bias=False),
                nn.BatchNorm2d(out_channels)
                )
        
        self.shortcut = nn.Sequential()
        #确保shortcut和处理后的数据有相同的维度
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                    #nn.MaxPool2d(kernel_size=3,stride=2),
                    nn.Conv2d(in_channels,out_channels,kernel_size=3,padding = 1,stride = stride,bias=False),
                    nn.BatchNorm2d(out_channels)
                    )
        
    def forward(self, x):
            return nn.ReLU(True)(self.residual_function(x) + self.shortcut(x))
        
    def getweight0(self):
#        for name,parameters in self.named_parameters():
#            print(name,':',parameters.size())
            
        return self.state_dict()['residual_function.0.weight'].data.clone()
    
    def getweight3(self):
            
        return self.state_dict()['residual_function.3.weight'].data.clone()

class Resnet(nn.Module):
    def __init__(self,block,num_classes=10):
        super().__init__()
        #这个应该是继承nn。Module的一个初始化，那个self和super里面的Resnet是啥意思
        
        #nn.Conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True))
        #BatchNorm2d(channel-number)
        #relu（true）ture会减少显存占用，但是会改变输入数据，详见https://blog.csdn.net/tmk_01/article/details/80679991
        self.conv1 = nn.Sequential(
                nn.Conv2d(3,64,kernel_size=7,stride=2,padding = (3,3),bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
                )
        
        self.conv2_x = self._make_layer(block,64,64)
        self.conv3_x = self._make_layer(block,64,128)
        self.conv4_x = self._make_layer(block,128,256)
        self.conv5_x = self._make_layer(block,256,512)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1)) #应该是把多维度的内容都变成一维度的平均值了吧
        
        self.fc = nn.Linear(512,num_classes)
    
    
    def _make_layer(self,block,in_channels,out_channels):
        layers = []
        layers.append(block(in_channels,out_channels,stride = 2))#在第一个block里面扩展维度，并且降低分辨率
        layers.append(block(out_channels,out_channels, stride = 1))#就是因为resnet18，每层有两个block，所以这边简写了，不然应该输入数组
        
        return nn.Sequential(*layers)
    
    def get_featureconv1(self,image):
        self.conv1.cpu()
        output = self.conv1(image)
        self.conv1.cuda()
        return output
    
    
    
    def get_reftconv1(self,image,num):
        m = self.conv1.cpu()
        m.eval()
        output = m(image)
        #得到64个16*16的tensor吧，然后分别取出来，把其他通道置0
        c_output = output.clone()
        c_output[:,0:num,:,:]=0
        c_output[:,num+1:,:,:] = 0
        q = nn.ReLU()
        c_output = q(c_output)
        weight = self.conv1[0].weight.data.clone()
        c_output = deconv(weight,c_output,2,64,3,7,3)
        #这个新的output
        self.conv1.cuda()
        return c_output
    
    def get_reftconv5(self,image,num):
        m = self.conv1.cpu()
        m.eval()
        output = m(image)
        m = self.conv2_x.cpu()
        m.eval()
        output = m(output)
        m = self.conv3_x.cpu()
        m.eval()
        output = m(output)
        m = self.conv4_x.cpu()
        m.eval()
        output = m(output)
        m = self.conv5_x.cpu()
        m.eval()
        output = m(output)
        
        c_output = output.clone()
        c_output[:,0:num,:,:]=0
        c_output[:,num+1:,:,:] = 0
        
        #5-》4
        q = nn.ReLU()
        c_output = q(c_output)
        weight = self.conv5_x[1].getweight3()
        c_output = deconv(weight,c_output,1,512,512,3,1)
        q = nn.ReLU()
        c_output = q(c_output)
        weight = self.conv5_x[1].getweight0()
        c_output = deconv(weight,c_output,1,512,512,3,1)
        print(c_output.size())
        q = nn.ReLU()
        c_output = q(c_output)
        weight = self.conv5_x[0].getweight3()
        c_output = deconv(weight,c_output,1,512,512,3,1)
        q = nn.ReLU()
        c_output = q(c_output)
        weight = self.conv5_x[0].getweight0()
        c_output = deconv(weight,c_output,2,512,256,3,1)
        print(c_output.size())
        
        #4-3
        q = nn.ReLU()
        c_output = q(c_output)
        weight = self.conv4_x[1].getweight3()
        c_output = deconv(weight,c_output,1,256,256,3,1)
        q = nn.ReLU()
        c_output = q(c_output)
        weight = self.conv4_x[1].getweight0()
        c_output = deconv(weight,c_output,1,256,256,3,1)
        q = nn.ReLU()
        c_output = q(c_output)
        weight = self.conv4_x[0].getweight3()
        c_output = deconv(weight,c_output,1,256,256,3,1)
        q = nn.ReLU()
        c_output = q(c_output)
        weight = self.conv4_x[0].getweight0()
        c_output = deconv(weight,c_output,2,256,128,3,1)
        
        #3-2
        q = nn.ReLU()
        c_output = q(c_output)
        weight = self.conv3_x[1].getweight3()
        c_output = deconv(weight,c_output,1,128,128,3,1)
        q = nn.ReLU()
        c_output = q(c_output)
        weight = self.conv3_x[1].getweight0()
        c_output = deconv(weight,c_output,1,128,64,3,1)
        q = nn.ReLU()
        c_output = q(c_output)
        weight = self.conv3_x[0].getweight3()
        c_output = deconv(weight,c_output,1,128,128,3,1)
        q = nn.ReLU()
        c_output = q(c_output)
        weight = self.conv3_x[0].getweight0()
        c_output = deconv(weight,c_output,2,128,64,3,1)
        
        #2-1
        q = nn.ReLU()
        c_output = q(c_output)
        weight = self.conv2_x[1].getweight3()
        c_output = deconv(weight,c_output,1,64,64,3,1)
        q = nn.ReLU()
        c_output = q(c_output)
        weight = self.conv2_x[1].getweight0()
        c_output = deconv(weight,c_output,1,64,64,3,1)
        q = nn.ReLU()
        c_output = q(c_output)
        weight = self.conv2_x[0].getweight3()
        c_output = deconv(weight,c_output,1,64,64,3,1)
        q = nn.ReLU()
        c_output = q(c_output)
        weight = self.conv2_x[0].getweight0()
        c_output = deconv(weight,c_output,2,64,64,3,1)
        
        #1-output
        q = nn.ReLU()
        c_output = q(c_output)
        weight = self.conv1[0].weight.data.clone()
        c_output = deconv(weight,c_output,2,64,3,7,3)
        
        #放到cpu的还要放回gpu
        self.conv1.cuda()
        self.conv2_x.cuda()
        self.conv3_x.cuda()
        self.conv4_x.cuda()
        self.conv5_x.cuda()
        return c_output
        
        
    def forward(self,x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0),-1) #就是reshape一下输出
        output = self.fc(output)
        
        return output
    
def deconv(weight,image,stride,in_channel,out_channel,kernelsize,pad):
        m = nn.ConvTranspose2d(in_channel,out_channel, kernelsize,stride=stride,padding=pad,output_padding=stride-1,bias=False)
        m.weight.data = weight
        m.eval()
        output = m(image)
        return output
    
def resnet():
    return Resnet(BasicBlock)