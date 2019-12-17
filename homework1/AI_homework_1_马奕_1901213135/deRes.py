import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import sys
import res18
from res18 import scut


# res_weight=[{},{},{},{}]
# scut={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0}
# 反向残差快，输出图像尺寸差距用output_padding填补
class deResBlock(nn.Module):
    def __init__(self, inchannels, outchannels, net, stride=2, idx=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1, bias=False),
            #            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inchannels, outchannels, kernel_size=3, stride=stride, padding=1,
                               output_padding=stride - 1, bias=False),
            #            nn.BatchNorm2d(outchannels)
        )
        self.shortcut = scut[idx]  # 保存的sc
        # print(self.shortcut.shape)
        self.init_weight(idx, net)
        # print(idx)

    def forward(self, x):
        out = nn.functional.relu(x)
        out -= self.shortcut
        out = self.block(out)
        return out

    def init_weight(self, idx, net):  # idx:0~3   ->  10 11 20 21
        i = idx // 2 + 1
        j = idx % 2
        self.block[0].weight.data = net.features[i][j].block[3].weight.data
        self.block[2].weight.data = net.features[i][j].block[0].weight.data


class deFirstlayer(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.first = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, output_padding=1, padding=3)
        )
        self.first[1].weight.data = net.features[0].first[0].weight.data

    def forward(self, x):
        out = self.first(x)

        return out


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


class deResNet18(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.features = nn.Sequential(
            deResBlock(512, 512, net=net, stride=1, idx=7),
            deResBlock(512, 256, net=net, stride=2, idx=6),
            deResBlock(256, 256, net=net, stride=1, idx=5),
            deResBlock(256, 128, net=net, stride=2, idx=4),
            deResBlock(128, 128, net=net, stride=1, idx=3),
            deResBlock(128, 64, net=net, stride=2, idx=2),
            deResBlock(64, 64, net=net, stride=1, idx=1),
            deResBlock(64, 64, net=net, stride=2, idx=0),
            deFirstlayer(net=net)
        )
        self.deconv_indices = {0: 8, 1: 6, 2: 4, 3: 2, 4: 0}  # index

    #    def forward(self, x):
    #        out = self.features(x)
    #        return out
    def forward(self, x, layer):
        if layer in self.deconv_indices:
            start_idx = self.deconv_indices[layer]
        else:
            raise ValueError('layer is not a conv feature map')

        for idx in range(start_idx, len(self.features)):
            x = self.features[idx](x)
        return x


# def dreslayer(inchannels, outchannels, stride, idx=0):
#    blk = [deResBlock(inchannels, inchannels, stride=1, idx),
#           deResBlock(inchannels, outchannels, stride=stride, idx)]
#    return nn.Sequential(*blk)
#
# def reslayer(inchannels, outchannels, stride):
#    blk = [ResBlock(inchannels, outchannels, stride=stride),
#           ResBlock(outchannels, outchannels, stride=1)]
#    return nn.Sequential(*blk)
#


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

        scut[idx] = 0
        self.idx = idx

    def forward(self, x):
        scut[self.idx] = self.shortcut(x)
        out = self.block(x)
        out += scut[self.idx]
        out = nn.functional.relu(out)
        return out


# de1=deResBlock(512, 512, stride=1, idx=7)
# de2=deResBlock(512, 256, stride=2, idx=6)
# de3=deResBlock(256, 256, stride=1, idx=5)
# de4=deResBlock(256, 128, stride=2, idx=4)
# de5=deResBlock(128, 128, stride=1, idx=3)
# de6=deResBlock(128, 64, stride=2, idx=2)
# de7=deResBlock(64, 64, stride=1, idx=1)
# de8=deResBlock(64, 64, stride=2, idx=0)
if __name__ == '__main__':
    # net=deFirstlayer()
    haha = deResNet18(res18.ResNet18())
    # dnet=deResBlock(512,256,idx=6)
    # net=res18.ResNet18()
    x = torch.rand(1, 256, 14, 14)
    print(x.size())
    y = haha(x, 3)
    print(y.size())
    # z=dnet(y)
    # print(z.size())
    # z=dnet(y)
    # print(z.size())
    hh = Firstlayer()
    he = deFirstlayer(net)
    a = torch.rand(1, 3, 224, 224)
    print(a.size())
    b = hh(a)
    print(b.size())
    c = he(b)
    print(c.size())
# parm = {}   #filterviewer
# for name, parameters in net.named_parameters():
#     parm[name] = parameters.detach().cpu().numpy()
# weight = parm['0.first.0.weight']
# view.plot_conv_weights(weight, 'conv{}'.format(1))
