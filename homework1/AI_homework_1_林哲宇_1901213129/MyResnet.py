import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """docstring for ResBlock"""    
    def __init__(self, inplanes, planes, stride=1):
        super(ResBlock, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )
        if stride != 1 or inplanes != planes:
            self.identity = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),nn.BatchNorm2d(planes))
        else:
            self.identity = nn.Sequential()

    def forward(self, x):
        output = self.residual(x)
        output += self.identity(x)
        output = F.relu(output)
        return output

class ResNet(nn.Module):
    """docstring for ResNet"""
    def __init__(self, ResBlock, num_classes = 10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_x = self.make_layer(ResBlock, 64, stride = 1)
        # To Think about the stride
        self.conv3_x = self.make_layer(ResBlock, 128, stride = 2)
        self.conv4_x = self.make_layer(ResBlock, 256, stride = 2)
        self.conv5_x = self.make_layer(ResBlock, 512, stride = 2)
        # self.avgpool = 
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, planes, stride):
        modules = [stride] + [1]
        # To think about stride
        layers = []
        for stride in modules:
            layers.append(block(self.inplanes, planes, stride))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        conv1_output = output
        # TO DO MAXPOOL        
        output = self.maxpool(output)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        conv5_x_output = output        
        # output = F.avg_pool2d(output, 4)
        output = self.avgpool(output)
        # To think about pool
        output = torch.flatten(output, 1)
        # output = output.view(output.size(0), -1)
        output = self.fc(output)
        # return output
        return output, conv1_output, conv5_x_output