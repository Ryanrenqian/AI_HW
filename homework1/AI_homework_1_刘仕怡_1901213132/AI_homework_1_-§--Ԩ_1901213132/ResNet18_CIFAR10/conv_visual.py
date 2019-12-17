# coding with UTF-8
# ******************************************
# *****CIFAR-10 with ResNet8 in Pytorch*****
# *****conv_visual.py                  *****
# *****Author：Shiyi Liu               *****
# *****Time：  Oct 22nd, 2019          *****
# ******************************************
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import torch
import torchvision
import torchvision.transforms as transforms
import train_classify
from torch.utils.data import DataLoader
from deconv_network import Deconv1
from deconv_network import Deconv_ResNet18
import warnings
warnings.filterwarnings("ignore")
from torchvision.transforms import ToPILImage
from PIL import Image


def get_max_factor(x):
    factors = set()
    for i in range(1, int(math.sqrt(x))+1):
        if x % i == 0:
            factors.add(int(i))
            factors.add(int(x//i))
    factors = sorted(factors)
    num = len(factors)
    if num % 2 == 0:
        return factors[int(num/2)-1], factors[int(num/2)]
    return factors[int(num//2)], factors[int(num//2)]


def nomalize_img(img):
    max_v = np.max(img)
    min_v = np.min(img)
    return int(max_v), int(min_v), ((img - min_v) / (max_v - min_v))


# visual weight in size（B, C, W, H)
def visual_in_grid(weight, name, path, split_channels=True, normalize=False):
    max_value = np.max(weight)
    min_value = np.min(weight)
    channels = range(weight.shape[1])
    num_filters = weight.shape[0]
    row, col = get_max_factor(num_filters)
    fig, axes = plt.subplots(row, col)
    if not os.path.exists(path):
        os.makedirs(path)
    if split_channels:
        for ch in channels:
            for i, ax in enumerate(axes.flatten()):
                img = weight[i, ch]
                if normalize:
                    # img = img / 2 + 0.5
                    max_value, min_value, img = nomalize_img(img)
                # max_v, min_v, img = nomalize_img(weight[i, ch, :, :])
                # ax.imshow(img, vmin=min_value, vmax=max_value, interpolation='nearest', cmap='seismic')
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])
                # ax.title('filter {}'.format(ch))
            plt.savefig(os.path.join(path, '{}-{}.png'.format(name, ch)), bbox_inches='tight')
            # plt.close()
    else:
        for i, ax in enumerate(axes.flatten()):
            max_v, min_v, img = nomalize_img(weight[i, :, :, :])
            img = weight[i]
            if normalize:
                # img = img / 2 + 0.5
                max_value, min_value, img = nomalize_img(img)

            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.title('filter {}'.format(ch))
        plt.savefig(os.path.join(path, '{}.png'.format(name)), bbox_inches='tight')

    plt.close()


def visual_filter(model, path):
    model_params = model.state_dict()
    for name, para in model_params.items():
        if ('conv1.0' in name ) and 'weight' in name and para.dim() == 4:
            plot_path = os.path.join(path, 'filter', name)
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
            weight = para.cpu().numpy()
            visual_in_grid(weight, name, plot_path, split_channels=True)


def load_deconv_model(model):
    model_params = model.state_dict()
    
    weight_1 = torch.rand(64, 3, 32, 32)
    for name, para in model_params.items():
        if ('conv1.0' in name) and 'weight' in name and para.dim() == 4:
            weight_1 = para.cpu().clone()
            print(weight_1.size())
            break

    demodel1 = Deconv1(inplanes=weight_1.size()[0], outplanes=weight_1.size()[1], conv_weight=model.output_weight()[0],
                       padding=1)
    demodel5 = Deconv_ResNet18(deconv_weight=model.output_weight(), deconv1=demodel1)
    return demodel1, demodel5


def extract_one_feature(feature, i):
    extract_feature = torch.unsqueeze(feature[:, i, :, :], 1)
    if i > 0:
        zero_feature1 = torch.zeros((feature.size()[0], i,                   feature.size()[2], feature.size()[3]))
    else:
        zero_feature1 = None
    if i < feature.size()[1] - 1:
        zero_feature2 = torch.zeros((feature.size()[0], feature.size()[1]-i-1, feature.size()[2], feature.size()[3]))
    else:
        zero_feature2 = None
    if 0 < i < (feature.size()[1] - 1):
        return torch.cat((zero_feature1, extract_feature, zero_feature2), 1)
    elif i <= 0:
        return torch.cat((extract_feature, zero_feature2), 1)
    else:
        return torch.cat((zero_feature1, extract_feature), 1)


def visual_feature_map(model, path, device):
    mean, std = train_classify.load_mestd()
    bz = 16
    visual_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    visual_img_transform = transforms.Compose([transforms.ToTensor()])
    visual_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=visual_transform)
    visual_img_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=visual_img_transform)
    visual_loader = DataLoader(dataset=visual_set, batch_size=bz)
    visual_img_loader = DataLoader(dataset=visual_img_set, batch_size=bz)

    plot_path = os.path.join(path, 'feature_map/')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    with torch.no_grad():
        for n, (img, label) in enumerate(visual_loader):
            img, label = img.to(device), label.to(device)
            img2, label2 = visual_img_loader.collate_fn([visual_img_loader.dataset[i] for i in range(n, n+bz)])
            
            img2 = img2.to(device)
            
            feature1 = model.output_feature_conv1(img)
            visual_in_grid(feature1.cpu().permute(1, 0, 2, 3).numpy(), 'conv1_feature_map', plot_path,
                           split_channels=True)
            # plt.close()
            feature5 = model.output_feature_conv5(img)
            visual_in_grid(feature5.cpu().permute(1, 0, 2, 3).numpy(), 'conv5_feature_map', plot_path,
                           split_channels=True)

            deconv1_model, deconv5_model = load_deconv_model(model)
            deconv1_model = deconv1_model.to(device)
            deconv5_model = deconv5_model.to(device)
            
            recon_img1_all = torch.zeros((feature1.size()[1], bz, img.size()[1], img.size()[2], img.size()[3]))
            for i in range(feature1.size()[1]):
                one_map = extract_one_feature(feature1.cpu(), i).to(device)
                
                recon_img1 = deconv1_model(one_map)
                recon_img1_all[i] = recon_img1
                
                recon_show = torch.cat((img2, recon_img1), 0)
                visual_in_grid(recon_show.cpu().permute(0, 2, 3, 1).numpy(), 'recons_img_conv1_fea{}'.format(i), plot_path,
                               split_channels=False)
                plt.close()

            visual_in_grid(recon_img1_all.cpu().permute(0, 1, 3, 4, 2).numpy(), 'recons_img1_all', plot_path,
                           split_channels=True, normalize=True)
            plt.close()

            recon_img5_all = torch.zeros((feature5.size()[1], bz, img.size()[1], img.size()[2], img.size()[3]))
            for i in range(feature5.size()[1]):
                one_map = extract_one_feature(feature5.cpu(), i).to(device)
                # print(one_map.size())
                recon_img5 = deconv5_model(one_map)
                recon_img5_all[i] = recon_img5
                # print('recon:{}'.format(recon_img1.size()))
                recon_show = torch.cat((img2, recon_img5), 0)
                visual_in_grid(recon_show.cpu().permute(0, 2, 3, 1).numpy(), 'recons_img_conv5_fea{}'.format(i),
                               plot_path,
                               split_channels=False)
                plt.close()

            visual_in_grid(recon_img5_all.cpu().permute(0, 1, 3, 4, 2).numpy(), 'recons_img5_all', plot_path,
                           split_channels=True, normalize=True)
            plt.close()

            break

