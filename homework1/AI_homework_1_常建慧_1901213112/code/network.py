import functools

import torch
import torch.nn as nn
import torch.optim
from torch.nn import init
from torch.optim import lr_scheduler


def conv3x3(inplanes, outplanes, stride=1):
    return nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def get_norm_layer(layer_type='batch'):
    if layer_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif layer_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif layer_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError(
            'normalization layer [%s] is not found' % layer_type)
    return norm_layer


def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(
            nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer


# 基本块
class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None, s=1):
        super(BasicBlock, self).__init__()
        layers = []
        # self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if s == 1:
            layers += [conv3x3(inplanes, inplanes)]
            if norm_layer is not None:
                layers += [norm_layer(inplanes)]
            layers += [nl_layer()]
            layers += [conv3x3(inplanes, outplanes)]
            if norm_layer is not None:
                layers += [norm_layer(outplanes)]
            layers += [nl_layer()]
            self.shortcut = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        elif s == 2:
            layers += [conv3x3(inplanes, inplanes)]
            if norm_layer is not None:
                layers += [norm_layer(inplanes)]
            layers += [nl_layer()]
            layers += [conv3x3(inplanes, outplanes, stride=2)]
            if norm_layer is not None:
                layers += [norm_layer(outplanes)]
            layers += [nl_layer()]
            self.shortcut = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=2, padding=0)
        else:
            assert 0
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.conv(x)
        out = x0 + self.shortcut(x)
        return out


class Resnet(nn.Module):
    def __init__(self, input_nc, num_classes=10, ndf=64, n_blocks=4, norm='batch',
                 nl='lrelu'):
        super(Resnet, self).__init__()
        norm_layer = get_norm_layer(norm)
        nl_layer = get_non_linearity(nl)

        conv_layers = [nn.Conv2d(input_nc, ndf, kernel_size=7, stride=2, padding=3, bias=False)]
        conv_layers += [norm_layer(ndf), nl_layer()]
        conv_layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]

        output_ndf = 0
        for n in range(0, n_blocks):
            input_ndf = ndf * (2 ** max(0, n - 1))
            output_ndf = ndf * (2 ** n)
            if n == 0:
                conv_layers += [BasicBlock(input_ndf, output_ndf, norm_layer, nl_layer, s=1)]
            else:
                conv_layers += [BasicBlock(input_ndf, output_ndf, norm_layer, nl_layer, s=2)]

        conv_layers += [nn.AdaptiveAvgPool2d((1, 1))]
        self.conv = nn.Sequential(*conv_layers)
        self.fc = nn.Linear(output_ndf, num_classes)

    def forward(self, x):
        x_conv = self.conv(x)
        x_flat = torch.flatten(x_conv, 1)
        out = self.fc(x_flat)
        output = nn.Softmax()(out)

        return output


class ExtractFirstLayer(nn.Module):
    # 只定义到第一个卷积层
    def __init__(self, input_nc, num_classes=10, ndf=64, n_blocks=4, norm='batch',
                 nl='lrelu'):
        super(ExtractFirstLayer, self).__init__()
        self.conv1 = nn.Conv2d(input_nc, ndf, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        x_conv1 = self.conv1(x)
        return x_conv1


class ExtractLastLayer(nn.Module):
    """
    定义到最后一个卷积层 输出feature map
    """
    def __init__(self, input_nc, num_classes=10, ndf=64, n_blocks=4, norm='batch',
                 nl='lrelu'):
        super(ExtractLastLayer, self).__init__()

        norm_layer = get_norm_layer(norm)
        nl_layer = get_non_linearity(nl)

        conv_layers = [nn.Conv2d(input_nc, ndf, kernel_size=7, stride=2, padding=3, bias=False)]
        conv_layers += [norm_layer(ndf), nl_layer()]
        conv_layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]

        max_pool = [nn.Conv2d(input_nc, ndf, kernel_size=7, stride=2, padding=3, bias=False),
                    norm_layer(ndf), nl_layer(),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                 return_indices=True)]
        self.max_pool = nn.Sequential(*max_pool)

        for n in range(0, n_blocks):
            input_ndf = ndf * (2 ** max(0, n - 1))
            output_ndf = ndf * (2 ** n)
            if n == 0:
                conv_layers += [BasicBlock(input_ndf, output_ndf, norm_layer, nl_layer, s=1)]
            else:
                conv_layers += [BasicBlock(input_ndf, output_ndf, norm_layer, nl_layer, s=2)]

        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        x_conv = self.conv(x)
        _, indices = self.max_pool(x)
        return x_conv, indices


def init_net(net, init_type='normal', gpu_ids=[0], gain=0.02):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        device = torch.device('cuda:0')
        net.to(device)

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or
                                     classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
    return net


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler
