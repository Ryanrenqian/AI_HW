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
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
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
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_x = self._make_layer(64, layers[0])
        self.conv3_x = self._make_layer(128, layers[1], stride=2)
        self.conv4_x = self._make_layer(256, layers[2], stride=2)
        self.conv5_x = self._make_layer(512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_class)

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
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels, stride),
                nn.BatchNorm2d(out_channels),
            )

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
        x = self.maxpool(x)

        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18():
    model = ResNet()
    return model


def evaluate(model, testloader, device):
    model.eval()
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct/total