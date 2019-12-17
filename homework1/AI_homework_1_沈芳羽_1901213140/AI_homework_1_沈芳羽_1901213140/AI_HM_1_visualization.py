import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
import os
import math
import numpy as np
import random
from matplotlib import pyplot as plt
from resnet import FirstLayer,LastLayer
from resnet import BackFirstLayer,BackLastLayer

def prime_powers(n):    #乘法分解（m*n）
    factors=set()
    for x in range(1,int(math.sqrt(n)+1)):
        if n%x==0:
            factors.add(x)
            factors.add(int(n//x))
    return sorted(factors)

def get_grid_dim(x):
    factors=prime_powers(x)
    i=int(len(factors)/2)
    if len(factors)%2 == 0:
        return factors[i],factors[i-1]
    return factors[i],factors[i]

def visualize_conv1():
    model = torch.load("./model/net_10000.pth")
    conv1 = model["conv1.weight"]
    conv1 = conv1.cpu().numpy()
    conv1 = conv1.transpose([0, 2, 3, 1])

    plt.figure()
    for i in range(64):
        plt.subplot(8, 8, i+1)
        data = conv1[i, :, :, :]
        data = (data - data.min()) / (data.max() - data.min())
        plt.imshow(data)
        plt.axis("off")
    plt.savefig(os.path.join("./plot", 'Filter-visualize.png'))
    plt.show()

def visualize_layer(layer_type,image):
    layer_type = layer_type.lower()
    if layer_type not in ("first", "last"):
        raise ValueError("Parameters Error!")

    device = torch.device("cuda:0")

    if layer_type == "first":
        network = FirstLayer()
    elif layer_type == "last":
        network = LastLayer()
    else:
        assert 0
    network.load_state_dict(torch.load("./model/net_10000.pth"), strict=False)
    network.to(device, torch.float32)

    network.eval()
    with torch.no_grad():
        if layer_type == "first":
            features= network(image)
        if layer_type == "last":
            features,_ = network(image)
        features = features.cpu().numpy()
        image = image.squeeze().cpu().numpy().transpose([1, 2, 0])
        image = (image - image.min()) / (image.max() - image.min())

    if layer_type == "first":
        height, width = 8, 8
    elif layer_type == "last":
        height, width = 16, 32
    else:
        assert 0

    plt.figure(1)
    plt.imshow(image)
    plt.savefig(os.path.join("./plot", '{}-src-feature.png'.format(layer_type)))
    plt.axis("off")
    plt.figure(2)
    for i in range(height * width):
        plt.subplot(height, width, i + 1)
        data = features[0, i, :, :]
        plt.imshow(data, cmap="gray")
        plt.axis("off")
    plt.savefig(os.path.join("./plot", '{}-featruemap.png'.format(layer_type)))
    plt.show()

def reconstruction(layer_type,image):
    layer_type = layer_type.lower()
    if layer_type not in ("first", "last"):
        raise ValueError("Parameters Error!")

    device = torch.device("cuda:0")

    if layer_type=="first":
        back_model = BackFirstLayer()
        back_model.to(device, torch.float32)

        network = FirstLayer()
        network.load_state_dict(torch.load("./model/net_10000.pth"), strict=False)
        network.to(device, torch.float32)

        with torch.no_grad():
            feature_1 = network(image)
            back_model.deconv1.weight = network.conv1.weight
            feature_recon = back_model(feature_1)
            fig, axes = plt.subplots(1, 2)
            recon_image=feature_recon[0]
            recon_image = recon_image.cpu().detach().numpy().transpose(1, 2, 0)
            recon_image = (recon_image - recon_image.min()) / (recon_image.max() - recon_image.min())
            axes[0].imshow(recon_image)
            image = image.squeeze().cpu().numpy().transpose([1, 2, 0])
            image = (image - image.min()) / (image.max() - image.min())
            axes[1].imshow(image)

            for i in range(2):
                axes[i].set_xticks([])
                axes[i].set_yticks([])
            plt.savefig(os.path.join("./plot", '{}-reconstruction.png'.format(layer_type)))
            plt.show()

    if layer_type == "last":
        network = LastLayer()
        network.load_state_dict(torch.load("./model/net_10000.pth"), strict=False)
        network.to(device, torch.float32)

        with torch.no_grad():
            feature_5,indices = network(image)
            back_model = BackLastLayer(indices)
            back_model.to(device, torch.float32)

            back_model.deblk11.deconv1.weight = network.blk42.conv2.weight
            back_model.deblk11.deconv2.weight = network.blk42.conv1.weight
            back_model.deblk12.deconv1.weight = network.blk41.conv2.weight
            back_model.deblk12.deconv2.weight = network.blk41.conv1.weight
            back_model.deblk12.extra[0].weight = network.blk41.extra[0].weight

            back_model.deblk21.deconv1.weight = network.blk32.conv2.weight
            back_model.deblk21.deconv2.weight = network.blk32.conv1.weight
            back_model.deblk22.deconv1.weight = network.blk31.conv2.weight
            back_model.deblk22.deconv2.weight = network.blk31.conv1.weight
            back_model.deblk22.extra[0].weight = network.blk31.extra[0].weight

            back_model.deblk31.deconv1.weight = network.blk22.conv2.weight
            back_model.deblk31.deconv2.weight = network.blk22.conv1.weight
            back_model.deblk32.deconv1.weight = network.blk21.conv2.weight
            back_model.deblk32.deconv2.weight = network.blk21.conv1.weight
            back_model.deblk32.extra[0].weight = network.blk21.extra[0].weight

            back_model.deblk41.deconv1.weight = network.blk12.conv2.weight
            back_model.deblk41.deconv2.weight = network.blk12.conv1.weight
            back_model.deblk42.deconv1.weight = network.blk11.conv2.weight
            back_model.deblk42.deconv2.weight = network.blk11.conv1.weight


            back_model.deconv1.weight = network.conv1.weight

            feature_recon = back_model(feature_5)
            fig, axes = plt.subplots(1, 2)
            recon_image = feature_recon[0]
            recon_image = recon_image.cpu().detach().numpy().transpose(1, 2, 0)
            recon_image = (recon_image - recon_image.min()) / (recon_image.max() - recon_image.min())
            axes[0].imshow(recon_image)
            image = image.squeeze().cpu().numpy().transpose([1, 2, 0])
            image = (image - image.min()) / (image.max() - image.min())
            axes[1].imshow(image)

            for i in range(2):
                axes[i].set_xticks([])
                axes[i].set_yticks([])
            plt.savefig(os.path.join("./plot", '{}-reconstruction.png'.format(layer_type)))
            plt.show()

def reconstruction_per_channel(layer_type,channels,image):
    layer_type = layer_type.lower()
    if layer_type not in ("first", "last"):
        raise ValueError("Parameters Error!")
    device = torch.device("cuda:0")


    if layer_type == "first":
        img = []

        back_model = BackFirstLayer()
        back_model.to(device, torch.float32)

        network = FirstLayer()
        network.load_state_dict(torch.load("./model/net_10000.pth"), strict=False)
        network.to(device, torch.float32)
        back_model.deconv1.weight = network.conv1.weight
        with torch.no_grad():
            feature_1 = network(image)
            feature_1_clone = feature_1.clone()
            # channels = feature_1.size(1)
            for channel in range(channels):
                temp = feature_1[:, channel, :, :].clone()
                feature_1_clone.zero_()
                feature_1_clone[:, channel, :, :] = temp
                feature_recon = back_model(feature_1_clone)  # tensor(batch,3,32,32)
                img.append(feature_recon)
        image = image.squeeze().cpu().numpy().transpose([1, 2, 0])
        image = (image - image.min()) / (image.max() - image.min())
        plt.figure(1)
        plt.imshow(image)
        plt.savefig(os.path.join("./plot", '{}-src-reconstruction-conv1.png'.format(layer_type)))
        plt.axis("off")
        plt.figure(2)
        grid_col, grid_row = get_grid_dim(channels)
        fig, axes = plt.subplots(grid_row, grid_col)
        axes_enum = axes.flat
        for l, ax in enumerate(axes_enum):
            recon_image = img[l][0]
            recon_image = recon_image.cpu().numpy().transpose(1, 2, 0)
            recon_image = (recon_image - recon_image.min()) / (recon_image.max() - recon_image.min())
            ax.imshow(recon_image, cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
        plt.savefig(os.path.join("./plot", '{}-reconstruction-perchannel.png'.format(layer_type)))
        plt.show()


    if layer_type == "last":
        img = []

        network= LastLayer()
        network.load_state_dict(torch.load("./model/net_10000.pth"), strict=False)
        network.to(device, torch.float32)
        with torch.no_grad():
            feature_5, indices = network(image)
        back_model = BackLastLayer(indices)
        back_model.to(device, torch.float32)
        back_model.deblk11.deconv1.weight = network.blk42.conv2.weight
        back_model.deblk11.deconv2.weight = network.blk42.conv1.weight
        back_model.deblk12.deconv1.weight = network.blk41.conv2.weight
        back_model.deblk12.deconv2.weight = network.blk41.conv1.weight
        back_model.deblk12.extra[0].weight = network.blk41.extra[0].weight

        back_model.deblk21.deconv1.weight = network.blk32.conv2.weight
        back_model.deblk21.deconv2.weight = network.blk32.conv1.weight
        back_model.deblk22.deconv1.weight = network.blk31.conv2.weight
        back_model.deblk22.deconv2.weight = network.blk31.conv1.weight
        back_model.deblk22.extra[0].weight = network.blk31.extra[0].weight

        back_model.deblk31.deconv1.weight = network.blk22.conv2.weight
        back_model.deblk31.deconv2.weight = network.blk22.conv1.weight
        back_model.deblk32.deconv1.weight = network.blk21.conv2.weight
        back_model.deblk32.deconv2.weight = network.blk21.conv1.weight
        back_model.deblk32.extra[0].weight = network.blk21.extra[0].weight

        back_model.deblk41.deconv1.weight = network.blk12.conv2.weight
        back_model.deblk41.deconv2.weight = network.blk12.conv1.weight
        back_model.deblk42.deconv1.weight = network.blk11.conv2.weight
        back_model.deblk42.deconv2.weight = network.blk11.conv1.weight

        back_model.deconv1.weight = network.conv1.weight
        with torch.no_grad():
            feature_5_clone = feature_5.clone()
            # channels = feature_5.size(1)
            for channel in range(channels):
                temp = feature_5[:, channel, :, :].clone()
                feature_5_clone.zero_()
                feature_5_clone[:, channel, :, :] = temp
                feature_recon = back_model(feature_5_clone)  # tensor(batch,3,32,32)
                img.append(feature_recon)
        image = image.squeeze().cpu().numpy().transpose([1, 2, 0])
        image = (image - image.min()) / (image.max() - image.min())
        plt.figure(1)
        plt.imshow(image)
        plt.savefig(os.path.join("./plot", '{}-src-reconstruction-conv5.png'.format(layer_type)))
        plt.axis("off")
        plt.figure(2)
        grid_col, grid_row = get_grid_dim(channels)
        fig, axes = plt.subplots(grid_row, grid_col)
        axes_enum = axes.flat
        for l, ax in enumerate(axes_enum):
            recon_image = img[l][0]
            recon_image = recon_image.cpu().numpy().transpose(1, 2, 0)
            recon_image = (recon_image - recon_image.min()) / (recon_image.max() - recon_image.min())
            ax.imshow(recon_image)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.savefig(os.path.join("./plot", '{}-reconstruction-perchannel.png'.format(layer_type)))
        plt.show()



if __name__ == "__main__":
    plotpath = "./plot"  
    isExists = os.path.exists(plotpath)
    if not isExists:
        os.makedirs(plotpath)
    tfs = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = CIFAR10(root='./data', train=False, download=True, transform=tfs)

    device = torch.device("cuda:0")
    image, _ = dataset[random.randrange(len(dataset))]
    image = image.to(device, torch.float32).unsqueeze(0)

    visualize_conv1()
    visualize_layer("first",image)
    visualize_layer("last",image)
    reconstruction("first",image)
    reconstruction("last",image)
    reconstruction_per_channel("first",9,image)
    reconstruction_per_channel("last",9,image)