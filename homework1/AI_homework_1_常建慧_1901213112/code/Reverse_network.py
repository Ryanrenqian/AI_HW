import torch.nn as nn


class Reverse_basic_block(nn.Module):
    def __init__(self, inplanes, outplanes, s=2):
        super(Reverse_basic_block, self).__init__()
        layers = []
        if s == 2:
            layers += [nn.ConvTranspose2d(outplanes, inplanes, 3, 2, 1, bias=False, output_padding=1)]
            layers += [nn.ReLU()]
            layers += [nn.ConvTranspose2d(inplanes, inplanes, 3, 1, 1, bias=False)]
            layers += [nn.ReLU()]
            self.shortcut = nn.ConvTranspose2d(outplanes, inplanes, 1, 2, 0, output_padding=1)
        elif s == 1:
            layers += [nn.ConvTranspose2d(outplanes, inplanes, 3, 1, 1, bias=False)]
            layers += [nn.ReLU()]
            layers += [nn.ConvTranspose2d(inplanes, inplanes, 3, 1, 1, bias=False)]
            layers += [nn.ReLU()]
            self.shortcut = nn.ConvTranspose2d(outplanes, inplanes, 1, 1, 0)
        else:
            assert 0
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.conv(x)
        out = x0 + self.shortcut(x)
        return out


class Deconv5_resnet(nn.Module):
    """
    Deconv the last layer
    """

    def __init__(self):
        super(Deconv5_resnet, self).__init__()
        deconv = []
        for n in reversed(range(4)):
            input_ndf = 64 * (2 ** max(0, n - 1))
            output_ndf = 64 * (2 ** n)
            if n == 0:
                deconv += [Reverse_basic_block(input_ndf, output_ndf, s=1)]
            else:
                deconv += [Reverse_basic_block(input_ndf, output_ndf, s=2)]
        self.max_unpool = nn.MaxUnpool2d(3, 2, 1)
        self.deconv_last = nn.Sequential(*[nn.ReLU(),
                                           nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3,
                                                              bias=False, output_padding=1)])
        self.deconv = nn.Sequential(*deconv)

    def forward(self, x, indices):
        out = self.deconv(x)
        out = self.max_unpool(out, indices=indices, output_size=[x.size()[0], 64, 112, 112])
        out = self.deconv_last(out)
        return out


class Deconv1_resnet(nn.Module):
    """
    Deconv the first layer
    """

    def __init__(self):
        super(Deconv1_resnet, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3, bias=False, output_padding=1)

    def forward(self, feature):
        out = self.deconv1(feature)
        return out
