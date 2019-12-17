import torch
from torch import nn
from torch.nn import functional as F


class ResBlk(nn.Module):
    """
    resnet block
    """

    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if stride!=1 or ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        out = F.relu(out)

        return out

class ResNet18(nn.Module):
    """
    resnet model
    """
    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1= nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,return_indices=True)

        # followed 4 blocks
        self.blk11 = ResBlk(64, 64, stride=1)
        self.blk12 = ResBlk(64, 64, stride=1)

        self.blk21 = ResBlk(64, 128, stride=2)
        self.blk22 = ResBlk(128, 128, stride=1)

        self.blk31 = ResBlk(128, 256, stride=2)
        self.blk32 = ResBlk(256, 256, stride=1)

        self.blk41 = ResBlk(256, 512, stride=2)
        self.blk42 = ResBlk(512, 512, stride=1)

        self.outlayer = nn.Linear(512 * 1 * 1, 10)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x,indices=self.pool(x)
        x = self.blk11(x)
        x = self.blk12(x)

        x = self.blk21(x)
        x = self.blk22(x)

        x = self.blk31(x)
        x = self.blk32(x)

        x = self.blk41(x)
        x = self.blk42(x)

        x = F.adaptive_avg_pool2d(x, [1, 1])
        # print('after pool:', x.shape)
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)

        return x,indices

class FirstLayer(nn.Module):
    """
    FirstLayer
    """

    def __init__(self):
        super(FirstLayer, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.conv1(x)
        x=self.bn1(x)
        x=F.relu(x)
        return x

class LastLayer(nn.Module):
    """
    LastLayer
    """

    def __init__(self):
        super(LastLayer, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)

        # followed 4 blocks
        self.blk11 = ResBlk(64, 64, stride=1)
        self.blk12 = ResBlk(64, 64, stride=1)

        self.blk21 = ResBlk(64, 128, stride=2)
        self.blk22 = ResBlk(128, 128, stride=1)

        self.blk31 = ResBlk(128, 256, stride=2)
        self.blk32 = ResBlk(256, 256, stride=1)

        self.blk41 = ResBlk(256, 512, stride=2)
        self.blk42 = ResBlk(512, 512, stride=1)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x, indices = self.pool(x)
        x = self.blk11(x)
        x = self.blk12(x)

        x = self.blk21(x)
        x = self.blk22(x)

        x = self.blk31(x)
        x = self.blk32(x)

        x = self.blk41(x)
        x = self.blk42(x)

        return x,indices



class BackResBlk(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(BackResBlk, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1)
        if ch_in == ch_out:
            self.deconv2 = nn.ConvTranspose2d(ch_in, ch_out, 3, stride=stride, padding=1)
        else:
            self.deconv2 = nn.ConvTranspose2d(ch_in, ch_out, 3, stride=stride, padding=1, output_padding=1)
        self.extra = nn.Sequential()

        if stride!=1 or ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.ConvTranspose2d(ch_in, ch_out, kernel_size=1, stride=stride,output_padding=1),
            )

    def forward(self, x):
        x_res = self.extra(x)
        x = F.relu(self.deconv1(x))
        x=self.deconv2(x)
        x = F.relu(x_res + x)

        return x


class BackFirstLayer(nn.Module):
    def __init__(self):
        super(BackFirstLayer, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3,output_padding=1)


    def forward(self, x):
        x = self.deconv1(x)
        # x = F.relu(x)
        return x

class BackLastLayer(nn.Module):
    def __init__(self,indices):
        super(BackLastLayer, self).__init__()
        self.deblk11 = BackResBlk(512,512,1)
        self.deblk12 = BackResBlk(512,256,2)

        self.deblk21 = BackResBlk(256,256,1)
        self.deblk22 = BackResBlk(256,128,2)

        self.deblk31 = BackResBlk(128,128,1)
        self.deblk32 = BackResBlk(128,64 ,2)

        self.deblk41 = BackResBlk(64, 64, 1)
        self.deblk42 = BackResBlk(64 ,64 ,1)

        self.max_unpool = nn.MaxUnpool2d(kernel_size=3, stride=2, padding=1)
        self.deconv1 = nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3,output_padding=1)

        self.indices = indices

    def forward(self, x):
        x = self.deblk11(x)
        x = self.deblk12(x)

        x = self.deblk21(x)
        x = self.deblk22(x)

        x = self.deblk31(x)
        x = self.deblk32(x)

        x = self.deblk41(x)
        x = self.deblk42(x)
        x = self.max_unpool (x,self.indices,output_size=torch.Size([x.size()[0], 64, 112, 112]))
        x = self.deconv1(x)
        x = F.relu(x)
        return x

