import torch
from torch import nn
import torch.nn.functional as F
class DeConv1(nn.Module):
    # Conv1 反卷积
    def __init__(self, in_channel, out_channel, kernel_size=(7, 7), stride=2, padding=3):
        super(DeConv1, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                                          padding=padding)
        self.init()

    def init(self, pth_path='runs/cifar10_resnet18_experiment_1/56_epoch_para.pkl'):
        resnet_state_dict = torch.load(pth_path)
        self.deconv.weight.data = resnet_state_dict['conv1.0.weight']

    def forward(self, featuremap):
        return self.deconv(featuremap)


class DeResBlock(nn.Module):
    def __init__(self,in_channel,out_channel,stride):
        super(DeResBlock,self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.deshortcut = nn.ConvTranspose2d(in_channel,out_channel,kernel_size=1,stride=stride,padding=0,bias=False)
        self.deleft = nn.Sequential(
            nn.ConvTranspose2d(in_channel,in_channel,kernel_size=3,stride=1,padding=1,bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channel,out_channel,kernel_size=3,stride=stride,padding=1,bias=False)
        )

    def forward(self, x):
        x = self.relu(x)
        # x = self.deshortcut(x)
        x = self.deleft(x)
        return x

class DeConv4(nn.Module):
    # 反卷积第五层
    def __init__(self):
        super(DeConv4,self).__init__()
        # self.delayer4 = DeResBlock(512,256,stride=2)
        self.delayer3 = DeResBlock(256,128, stride=2)
        self.delayer2 = DeResBlock(128,64,stride=2)
        self.delayer1 = DeResBlock(64,64,stride=2)
        self.derelu = nn.ReLU(inplace=True)
        self.conv1 = DeConv1(64,3,kernel_size=(7,7),stride=2,padding=3)
        self.init()


    def forward(self, x):
        # x = self.delayer4(x)
        x = self.delayer3(x)
        x = self.delayer2(x)
        x = self.delayer1(x)
        x = self.derelu(x)
        x = self.conv1(x)
        return x

    def init(self,pth_path='runs/cifar10_resnet18_experiment_1/56_epoch_para.pkl'):
        resnet_state_dict = torch.load(pth_path)
        self.conv1.deconv.weight.data = resnet_state_dict['conv1.0.weight']
        self.delayer1.deshortcut.weight.data = resnet_state_dict['layer1.0.shortcut.0.weight']
        self.delayer2.deshortcut.weight.data = resnet_state_dict['layer2.0.shortcut.0.weight']
        self.delayer3.deshortcut.weight.data = resnet_state_dict['layer3.0.shortcut.0.weight']
        # self.delayer4.deshortcut.weight.data = resnet_state_dict['layer4.0.shortcut.0.weight']
        self.delayer1.deleft[0].weight.data = resnet_state_dict['layer1.0.left.3.weight']
        self.delayer1.deleft[2].weight.data = resnet_state_dict['layer1.0.left.0.weight']
        self.delayer2.deleft[0].weight.data = resnet_state_dict['layer2.0.left.3.weight']
        self.delayer2.deleft[2].weight.data = resnet_state_dict['layer2.0.left.0.weight']
        self.delayer3.deleft[0].weight.data = resnet_state_dict['layer3.0.left.3.weight']
        self.delayer3.deleft[2].weight.data = resnet_state_dict['layer3.0.left.0.weight']
        # self.delayer4.deleft[0].weight.data = resnet_state_dict['layer4.0.left.3.weight']
        # self.delayer4.deleft[2].weight.data = resnet_state_dict['layer4.0.left.0.weight']


def plot_reconstruction(conv1_feature, conv4_feature, device):
    conv1_deconv = DeConv1(64, 3, kernel_size=(7, 7), stride=2, padding=3)
    conv1_deconv.to(device)
    conv1_reconstruction = conv1_deconv(conv1_feature)
    print(conv1_reconstruction.shape)
    # print(conv1_reconstruction)

    conv4_deconv = DeConv4()
    conv4_deconv.to(device)
    conv4_deconstruction = conv4_deconv(conv4_feature)
    print(conv4_deconstruction.shape)
    return conv1_reconstruction, conv4_deconstruction