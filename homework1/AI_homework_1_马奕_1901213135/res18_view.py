import torch
import torch.nn as nn
import numpy as np
from functools import partial
import res18
import deRes
import torchvision
import torchvision.transforms as transf
import matplotlib.pyplot as plt
import view_lib
import utils
import os

def store(model):  # 为每层输出加hook,保存特征图
    """
    make hook for feature map
    """

    def hook(module, input, output, key):
        if isinstance(module, nn.MaxPool2d):
            model.feature_maps[key] = output[0]
            model.pool_locs[key] = output[1]
        else:
            model.feature_maps[key] = output

    for idx, layer in enumerate(model._modules.get('features')):
        # _modules returns an OrderedDict
        layer.register_forward_hook(partial(hook, key=idx))


def vis_layer(layer, res_conv, res_deconv):  # 找到最大激活的featuremap，并通过反向网络重建
    """
    visualing the layer deconv result
    """
    num_feat = res_conv.feature_maps[layer].shape[1]  # 特征图个数

    # set other feature map activations to zero
    new_feat_map = res_conv.feature_maps[layer].clone()

    # choose the max activations map
    act_lst = []
    # 所有特征图取最大值
    for i in range(0, num_feat):
        choose_map = new_feat_map[0, i, :, :]
        activation = torch.max(choose_map)
        act_lst.append(activation.item())

    act_lst = np.array(act_lst)

    mark = np.argmax(act_lst)  # 返回最大激活值的位置
    # print(mark)
    choose_map = new_feat_map[0, mark, :, :]
    #    max_activation = torch.max(choose_map)

    # make zeros for other feature maps
    if mark == 0:
        new_feat_map[:, 1:, :, :] = 0
    else:
        new_feat_map[:, :mark, :, :] = 0
        if mark != res_conv.feature_maps[layer].shape[1] - 1:
            new_feat_map[:, mark + 1:, :, :] = 0

    #    choose_map = torch.where(choose_map == max_activation,  # 最大特征图中除了最大值像素，其他清零。 why? 反卷积输入应该是特征图啊
    #                             choose_map,
    #                             torch.zeros(choose_map.shape)
    #                             )

    # make zeros for ther activations
    new_feat_map[0, mark, :, :] = choose_map

    # print(torch.max(new_feat_map[0, mark, :, :]))
    #    print(max_activation)

    deconv_output = res_deconv(new_feat_map, layer)

    new_img = deconv_output.data.numpy()[0].transpose(1, 2, 0)  # (H, W, C)
    #    new_img=cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY)
    #    print(new_img.shape)
    #    aa=new_img
    # normalize
    new_img = (new_img - new_img.min()) / (new_img.max() - new_img.min()) * 255
    new_img = new_img.astype(np.uint8)
    # cv2.imshow('reconstruction img ' + str(layer), new_img)
    # cv2.waitKey()
    return new_img, int(mark)


Test = transf.Compose([
    transf.Resize(224, interpolation=4),
    transf.ToTensor(),
    transf.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if __name__ == '__main__':

    test_data = torchvision.datasets.CIFAR10(root='../data', train=False, transform=Test, download=False)
    onepic = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=0)

    cnt = 0
    for image, label in onepic:
        if cnt >= 1:
            break
        cnt += 1
        #        print(image.shape)
        # print(label)
        orimg = image.data.numpy()[0].transpose(1, 2, 0)

        net = res18.ResNet18()
        net.load_state_dict(torch.load("./net/resnet18.pth"))
        net.eval()
        store(net)
        conv_output = net(image)

        # backward processing
        denet = deRes.deResNet18(net)
        denet.eval()
        plt.figure(num=None, figsize=(16, 12), dpi=80)
        plt.subplot(2, 3, 1)
        plt.title('original picture')
        orimg = (orimg - orimg.min()) / (orimg.max() - orimg.min()) * 255
        orimg = orimg.astype(np.uint8)
        #        orimg = orimg.astype(np.uint8)
        plt.imshow(orimg)
        for idx, layer in enumerate([0, 1, 2, 3, 4]):
            # for idx, layer in enumerate(vgg16_conv.conv_layer_indices):
            plt.subplot(2, 3, idx + 2)
            img, activation = vis_layer(layer, net, denet)
            plt.title(f'restruction from Conv_{layer}, the max activations is {activation}')
            plt.imshow(img)

        # plt.show()
        utils.prepare_dir('./out_visualization', empty=False)
        plt.savefig('./out_visualization/restruction.jpg')
        # plt.savefig(os.path.join(PLOT_DIR, 'restruction.jpg'), bbox_inches='tight')
        # 显示第一层滤波器
        viewlayer = 0
        parm = {}  # filterviewer
        parmidx = {0: 0, 1: 16, 2: 34, 3: 52, 4: 70}
        for name, parameters in net.named_parameters():
            parm[name] = parameters.detach().cpu().numpy()
        parmdic = list(parm.keys())
        xx = parmdic[parmidx[viewlayer]]
        weight = parm[parmdic[parmidx[viewlayer]]]
        view_lib.plot_conv_weights(weight, 'conv{}'.format(viewlayer))

        # 显示特征图
        viewlayer = 0
        view_lib.plot_conv_output(net.feature_maps[viewlayer].detach().numpy(), 'layer{}'.format(viewlayer))
