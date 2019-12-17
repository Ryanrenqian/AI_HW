# coding with UTF-8
# ******************************************
# *****CIFAR-10 with ResNet8 in Pytorch*****
# *****network.py                      *****
# *****Author：Shiyi Liu               *****
# *****Time：  Oct 22nd, 2019          *****
# ******************************************
import torch
import torch.nn as nn
# import deconv_network


class BasicBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, groups=1, base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != outplanes:
            print()
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(outplanes)
            )
        self.stride = stride

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet18(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv2_x = self._make_layer(64, stride=1)
        self.conv3_x = self._make_layer(128, stride=2)
        self.conv4_x = self._make_layer(256, stride=2)
        self.conv5_x = self._make_layer(512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer(self, planes, stride):
        strides = [stride] + [1]
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.inplanes, planes, stride))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)

        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    # for feature visualization
    def output_feature_conv1(self, x):
        # out = self.conv1[0](x)
        # out = self.conv1[2](out)
        out = self.conv1(x)
        return out

    def output_feature_conv5(self, x):
        out = self.conv1(x)

        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        return out

    def output_weight(self):
        output_weights = [self.conv1[0].weight.data,
                          # conv2_x
                          self.conv2_x[0].conv1.weight.data,self.conv2_x[0].conv2.weight.data,
                          self.conv2_x[1].conv1.weight.data,self.conv2_x[1].conv2.weight.data,
                          # conv3_x
                          self.conv3_x[0].conv1.weight.data,
                          self.conv3_x[0].conv2.weight.data, self.conv3_x[0].shortcut[0].weight.data,
                          self.conv3_x[1].conv1.weight.data, self.conv3_x[1].conv2.weight.data,
                          # conv4_x
                          self.conv4_x[0].conv1.weight.data, self.conv4_x[0].conv2.weight.data,
                          self.conv4_x[0].shortcut[0].weight.data, self.conv4_x[1].conv1.weight.data,
                          self.conv4_x[1].conv2.weight.data,
                          #conv5_x
                          self.conv5_x[0].conv1.weight.data,
                          self.conv5_x[0].conv2.weight.data, self.conv5_x[0].shortcut[0].weight.data,
                          self.conv5_x[1].conv1.weight.data, self.conv5_x[1].conv2.weight.data]

        return output_weights
