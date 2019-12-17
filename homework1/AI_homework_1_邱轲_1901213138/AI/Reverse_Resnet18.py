import torch.nn as nn


# For the Reconstruction map of Feature map
class Reverse_Conv(nn.Module):
    def __init__(self, in_channel, out_channel, Conv_weight, kernel_size, stride, padding, output_padding):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.DeConv1 = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding,bias=False)
        # set weights from forword net
        self.DeConv1.weight.data = Conv_weight

    def forward(self, x):
        output = self.relu(x)
        output = self.DeConv1(output)
        return output

class Deconv_BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, block_weights, stride=1):
        super(Deconv_BasicBlock, self).__init__()
        self.Deconv2 = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.Deconv2.weight.data = block_weights[0]
        self.relu = nn.ReLU(inplace=True)
        self.Deconv1 = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, output_padding=stride-1, bias=False)
        self.Deconv1.weight.data = block_weights[1]
        self.shortcut = nn.Sequential()
        if in_channel != out_channel or stride != 1:
            self.shortcut = nn.Sequential(
                 nn.ConvTranspose2d(in_channel, out_channel,  kernel_size=1, stride=stride, output_padding=stride-1, padding=0, bias=False)
                 )
            self.shortcut[0].weight.data = block_weights[2]

    def forward(self, x):
        x = self.relu(x)
        out = self.relu(self.Deconv2(x))
        out = self.Deconv1(out)
        out += self.shortcut(x)
        return out

class Reverse_ResNet18(nn.Module):
    def __init__(self, weights, indices=None):
        self.in_channel = 512
        super(Reverse_ResNet18, self).__init__()
        self.Deconv5_x = self._make_layer(256, weights[15:20], 2)
        self.Deconv4_x = self._make_layer(128, weights[10:15], 2)
        self.Deconv3_x = self._make_layer(64, weights[5:10], 2)
        self.Deconv2_x = self._make_layer(64, weights[1:5], 1)
        self.Maxunpool = nn.MaxUnpool2d(kernel_size=3, stride=2, padding=1)
        self.indices = indices
        self.Deconv1 = Reverse_Conv(in_channel=16, out_channel=32, Conv_weight=weights[0], kernel_size=7, stride=2, padding=3, output_padding=1)

    def _make_layer(self, out_channel, block_weights, stride):
        layers = []
        if stride == 2:
            block_weight2 = [block_weights[4], block_weights[3]]
            block_weight1 = [block_weights[1], block_weights[0], block_weights[2]]
        else:
            block_weight2 = [block_weights[3], block_weights[2]]
            block_weight1 = [block_weights[1], block_weights[0]]

        layers.append(Deconv_BasicBlock(self.in_channel, self.in_channel, block_weight2, stride=1))
        layers.append(Deconv_BasicBlock(self.in_channel, out_channel, block_weight1, stride=stride))
        self.in_channel = out_channel
        return nn.Sequential(*layers)

    def forward(self, x, indices):
        output = self.Deconv5_x(x)
        output = self.Deconv4_x(output)
        output = self.Deconv3_x(output)
        output = self.Deconv2_x(output)
        output = self.Maxunpool(output, indices=indices, output_size=[x.size()[0], 64, 16, 16])
        output = self.Deconv1(output)
        return output
