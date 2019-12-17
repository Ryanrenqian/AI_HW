import torch.nn as nn
import torch


def conv3x3(in_channels, out_channels, stride=1):
    """
    3x3卷积层，并且隐藏了3x3卷积输入输出维度相同的条件
    :param in_channels:输入的通道数
    :param out_channels:输出通道数
    :param stride:卷积步长
    :return:创建好的3x3卷积
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, bias=False, padding=1)


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


def deconv3x3(in_channels, out_channels, stride=1, kernel_size=3):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=int((kernel_size - 1) / 2), output_padding=stride - 1, bias=False)


def deconv1x1(in_channels, out_channels, stride=1, kernel_size=1):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=int((kernel_size - 1) / 2), output_padding=stride - 1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        :param in_channels:输入通道数量
        :param out_channels:输出通道数量
        :param stride:步长
        """
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)  # inplace参数用来指示是否覆盖原变量，可用来减少内存占用
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        # identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, layers=[2, 2, 2, 2], num_class=10):
        """
        :param layers:定义每个layer的block数量，默认与Assignment 1中对应
        :param num_class:最终分类的类数
        :param groups:分组卷积组数
        """
        super(ResNet, self).__init__()

        self.in_channels = 64

        # 这段代码过后特征图已经为原来的1/4
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)  # inplace参数用来指示是否覆盖原变量，可用来减少内存占用
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)

        self.conv2_x = self._make_layer(64, layers[0])
        self.conv3_x = self._make_layer(128, layers[1], stride=2)
        self.conv4_x = self._make_layer(256, layers[2], stride=2)
        self.conv5_x = self._make_layer(512, layers[3], stride=2)

        # self.conv5_x_0_conv1 = conv3x3(256, 512, stride=2)
        # self.conv5_x_0_bn1 = nn.BatchNorm2d(512)
        # self.conv5_x_0_conv2 = conv3x3(512,512)
        # self.conv5_x_0_bn2 = nn.BatchNorm2d(512)
        #
        # self.conv5_x_1_conv1 = conv3x3(512, 512, stride=2)
        # self.conv5_x_1_bn1 = nn.BatchNorm2d(512)
        # self.conv5_x_1_conv2 = conv3x3(512, 512)

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512, num_class)

    def _make_layer(self, out_channels, blocks, stride=1):
        """
        用于创建Resnet的层，比如con1,con2-x,con3-x...
        :param block:指定创建的block类型
        :param out_channels:输出通道数
        :param blocks:block个数
        :param stride:步长大小默认为1
        :return:
        """
        downsample = None
        # if stride != 1 or self.in_channels != out_channels:
        #     downsample = nn.Sequential(
        #         conv1x1(self.in_channels, out_channels, stride),
        #         nn.BatchNorm2d(out_channels),
        #     )
        #
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x, _ = self.maxpool(x)

        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        return x


class deconvBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(deconvBasicBlock, self).__init__()
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv2 = deconv3x3(in_channels, in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = deconv3x3(in_channels, out_channels, stride)
        self.stride = stride

    def forward(self, feature):
        identity = feature
        # print("indetity size:", identity.size())
        out = self.bn2(feature)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.conv1(out)
        # print("out size:", out.size())
        # out -= identity
        out = self.relu(out)
        return out


class deoconvResNet(nn.Module):
    # pool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1,return_indices=True)
    def __init__(self, layers=[2, 2, 2, 2]):
        """
        :param layers:定义每个layer的block数量，默认与Assignment 1中对应
        :param num_class:最终分类的类数
        :param groups:分组卷积组数
        """
        super(deoconvResNet, self).__init__()

        self.in_channels = 512
        self.conv5_x = self._make_layer(256, layers[3], stride=2)
        self.conv4_x = self._make_layer(128, layers[2], stride=2)
        self.conv3_x = self._make_layer(64, layers[1], stride=2)
        self.conv2_x = self._make_layer(64, layers[0])
        self.maxunpool = nn.MaxUnpool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)  # inplace参数用来指示是否覆盖原变量，可用来减少内存占用
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3, bias=False)

    def _make_layer(self, out_channels, blocks, stride=1):
        """
        用于创建Resnet的层，比如con1,con2-x,con3-x...
        :param block:指定创建的block类型
        :param out_channels:输出通道数
        :param blocks:block个数
        :param stride:步长大小默认为1
        :return:
        """
        layers = []
        layers.append(deconvBasicBlock(self.in_channels, self.in_channels, stride))

        for _ in range(1, blocks):
            layers.append(deconvBasicBlock(self.in_channels, out_channels))
        self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv5_x(x)
        x = self.conv4_x(x)
        x = self.conv3_x(x)
        x = self.conv2_x(x)
        x = self.maxunpool(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.conv1(x)

        return x


class deconv_layer1(nn.Module):
    def __init__(self):
        super(deconv_layer1, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self,x):
        return self.deconv1(x)


def resnet18_without_fc():
    model = ResNet()
    return model


def deconv_resnet18_without_fc():
    model = deoconvResNet()
    return model
