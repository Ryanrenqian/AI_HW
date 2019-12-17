import torch
import torchvision
import torch.utils.data
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
import math
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from Reverse_Resnet18 import Reverse_Conv, Reverse_ResNet18
import warnings
warnings.filterwarnings('ignore')

Cifar10_train_mean = (0.4914, 0.4822, 0.4465)
Cifar10_train_std = (0.2023, 0.1994, 0.2010)
transform_visual = transforms.Compose([transforms.ToTensor(), transforms.Normalize(Cifar10_train_mean, Cifar10_train_std)])
visual_dataset = torchvision.datasets.CIFAR10(root='./data/', train=False, download=False, transform=transform_visual)
# One image per batch!
visual_batchsize = 1
visual_loader = torch.utils.data.DataLoader(dataset=visual_dataset, batch_size=visual_batchsize, shuffle=False)
device = torch.device('cuda')

def Filter_visualization(model, path1, path2,current_step):
    writer = SummaryWriter(log_dir=path1)
    for filter_key in model.state_dict():
        if 'Conv1.0.weight' in filter_key:

            # Get filter tensor
             Conv1_filter = model.state_dict()[filter_key]

             # show filter size
             x = torchvision.utils.make_grid(Conv1_filter, nrow = 8, padding = 2, normalize=False)  # 1*输入通道数, w, h
             writer.add_image('Conv1_filter', x, global_step=current_step)

             # Filter visualization based on Plot
             weights = Conv1_filter.cpu().clone()
             weights = weights.permute(3, 2, 1, 0) # [B, C, H, W] ->[H, W ,C ,B]
             weights = weights.numpy()
             Plot_visual(weights, path2+'Filter_visualization', 'Conv{}_filter_visualization'.format(1))
             plt.close()
             print('Filter Visualization Succeeded!')

    writer.close()

def Featuremap_visualization(model, path):

    model.eval()
    with torch.no_grad():
        # Initial Feature Map
        for batch_num, (visual_data, label) in enumerate(visual_loader):
            # if batch_num == 1:
                visual_data, label = visual_data.to(device), label.to(device)

                # [B, 64, 16, 16]
                Initial_map1 = model.visual_conv1(visual_data)
                Initial_map5, indices = model.visual_conv5(visual_data)

                # [64, B, 16, 16]
                Initial_map1 = Initial_map1.permute(1, 0, 2, 3)
                Initial_map5 = Initial_map5.permute(1, 0, 2, 3)
                data_img = visual_data.cpu().clone().permute(1, 0, 2, 3)

                # Make grid [3, 164, 164]
                Map1 = torchvision.utils.make_grid(Initial_map1, nrow=8, padding=4, normalize=True)
                Map5 = torchvision.utils.make_grid(Initial_map5, nrow=32, padding=1, normalize=True)


                # show and save
                for i in range(visual_batchsize):
                     plt_show_save(data_img[:, i, :, :], path+'Initial_Feature_Visualization', 'Initial visual_data[3 X 32 X 32 ]')
                plt_show_save(Map1, path+'Initial_Feature_Visualization', 'Initial Conv1 Feature map[64 X 16 X 16]')
                plt_show_save(Map5, path+'Initial_Feature_Visualization', 'Initial Conv5_x Feature map[512 X 1 X 1]')

                # Reconstruct Feature Map
                for filter_key in model.state_dict():
                    if 'Conv1.0.weight' in filter_key:
                         # Get weights tensor from Resnet18
                         Conv1_weights = model.state_dict()[filter_key]

                 # Conv1_weights -->[64, 3, 7, 7]
                Reverse_model1 = Reverse_Conv(in_channel=Conv1_weights.size()[0], out_channel=Conv1_weights.size()[1], Conv_weight=Conv1_weights,
                                               kernel_size=7, stride=2, padding=3, output_padding=1).to(device)

                # prepara for one image
                ALL_map1 = torch.zeros(64, 3, 32, 32)
                # [64, B, 16, 16] --> [B, 64, 16, 16]  [512, B, 1, 1] --> [B, 512, 1, 1]
                Initial_map1 = Initial_map1.permute(1, 0, 2, 3)
                Initial_map5 = Initial_map5.permute(1, 0, 2, 3)

                # Use full feature map to reconstruct image
                result = Reverse_model1(Initial_map1)
                result1 = result.cpu().clone()
                plt_show_save(result1[0, :, :, :], path+'Reconstruct_Feature_Visualization', 'Full channel reconstruction')

                for index in range(Initial_map1.size()[1]):
                    # for batch , per map --> get [1, 64, 16 ,16] to model
                    Map_index = get_onechannel_Map(Initial_map1.cpu(), index).to(device)

                    # for batch , per map --> [B, 3, 32, 32]
                    Deconv1_img = Reverse_model1(Map_index)
                    ALL_map1[index] = Deconv1_img[0]

                # show map with grid F
                Map_All = ALL_map1.cpu().clone()
                Map_All = Map_All.permute(2, 3, 1, 0)  # [B, C, H, W ] ->[H, W ,C ,B]  = [32, 32, 3, 64]
                Map_All = Map_All.numpy()
                Map_All = Map_All/2 + 0.5
                Plot_visual(Map_All, path+'Reconstruct_Feature_Visualization', 'Reconstruct_Featuremap_Conv1', False)

                # Get whole Reverse_model
                weights = get_all_weights(model)
                Reverse_model = Reverse_ResNet18(weights=weights).to(device)
                ALL_map2 = torch.zeros(512, 3, 32, 32)
                for index in range(Initial_map5.size()[1]):
                    Map_index = get_onechannel_Map(Initial_map5.cpu(), index).to(device)
                    Deconv_img = Reverse_model(Map_index, indices)
                    ALL_map2[index] = Deconv_img[0]

                # show map with grid F
                Map_All = ALL_map2.cpu().clone()
                Map_All = Map_All.permute(2, 3, 1, 0)  # [B, C, H, W ] ->[H, W ,C ,B]  = [32, 32, 3, 512]
                Map_All = Map_All.numpy()
                Map_All = Map_All/2 + 0.5
                Plot_visual(Map_All, path+'Reconstruct_Feature_Visualization', 'Reconstruct_Featuremap_Conv5', False)
                print('Feature Visualization Succeeded!')

                break

# Referred to the Conviz's Code
def Plot_visual(filter_weights, visual_path, name ,channels_all=True):

    # get and check path
    visual_path = os.path.join(visual_path, name)
    check_path(visual_path)
    min_weight = np.min(filter_weights)
    max_weight = np.max(filter_weights)

    # [H,  W , C, B] input
    channels = [0]
    # Set channels = 3
    if channels_all:
        channels = range(filter_weights.shape[2])

    # Set filter numbers = 64, get axes
    filters_num = filter_weights.shape[3]
    grid_r, grid_c = get_factors(filters_num)
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))
    if channels_all:
     # divided to 3 channel
        for channel in channels:
            # iterate filters inside every channel
             for index, Image in enumerate(axes.flat):
                  # get a single filter [H, W]
                 img = filter_weights[:, :, channel, index]
                  # put it on the [8 X 8] big grid
                 Image.imshow(img, vmin=min_weight, vmax=max_weight, interpolation='nearest', cmap='seismic')
                  # remove axes
                 Image.set_xticks([])
                 Image.set_yticks([])
             plt.savefig(os.path.join(visual_path, '{}-{}.png'.format(name, channel)), bbox_inches='tight')
    else:
        for index, Image in enumerate(axes.flat):
            img = filter_weights[:, :, :, index]
            Image.imshow(img, vmin=min_weight, vmax=max_weight, interpolation='nearest', cmap='seismic')
            Image.set_xticks([])
            Image.set_yticks([])
        plt.savefig(os.path.join(visual_path, '{}.png'.format(name)), bbox_inches='tight')


# show and save Image based on plt
def plt_show_save(tensor, path, name):
    if not os.path.exists(path):
        os.makedirs(path)
    tensor1 = tensor.cpu().clone()
    tensor1 = tensor1.numpy()
    np1 = np.transpose(tensor1, (1, 2, 0))
    # reverse Normalize
    max_v = np.max(np1)
    min_v = np.min(np1)
    np2 = (np1 - min_v) / (max_v - min_v)
    plt.title(name)
    plt.imshow(np2)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(path, '{}.png'.format(name)))
    #plt.show()

# Only one channel's Feature map is reserved, and the rest is set to 0
def get_onechannel_Map(Feature , index):

    B = Feature.size()[0]
    C = Feature.size()[1]
    H = Feature.size()[2]
    W = Feature.size()[3]
    Map_index = torch.unsqueeze(Feature[:, index, :, :], 1)
    #Map_index = torch.index_select(Feature, 0, torch.LongTensor([index]))
    if index == 0:
        return torch.cat((Map_index, torch.zeros((B, C-1, H, W))), 1)
    elif index == C-1:
        return torch.cat((torch.zeros((B, C - 1, H, W)), Map_index), 1)
    else:
        return torch.cat((torch.zeros((B, index, H, W)), Map_index, torch.zeros((B, C-index-1, H, W))), 1)

def get_all_weights(model):
    weights = []
    for filter_k, filter_w in model.state_dict().items():
        if 'Conv' in filter_k and 'weight' in filter_k and filter_w.dim() == 4:
            weights.append(filter_w)
    return weights


# check the path to store visualization
def check_path(visual_path):
    if not os.path.exists(visual_path):
        os.makedirs(visual_path)
    for the_file in os.listdir(visual_path):
        file_path = os.path.join(visual_path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Warning: {}'.format(e))

# get two max factors, Eg: given 12  --> 3 and 4
def get_factors(x):
    factors = set()
    for factor1 in range(1, int(math.sqrt(x)) + 1):
        if x % factor1 == 0:
            factors.add(int(factor1))
            factors.add(int(x // factor1))
    factors = sorted(factors)

    if len(factors) % 2 == 0:
        i = int(len(factors) / 2)
        return factors[i], factors[i - 1]
    i = len(factors) // 2
    return factors[i], factors[i]
