import torch
import numpy as np
from network import ExtractFirstLayer, ExtractLastLayer, Resnet
from Reverse_network import Deconv1_resnet, Deconv5_resnet
from matplotlib import pyplot as plt
from torchvision import transforms, datasets
import random


def visualize_conv1_filter():
    model = torch.load("./checkpoints/resnet/latest_net_resnet.pth")
    conv = model["conv.0.weight"]
    conv = conv.numpy()  # [64,3,7,7]
    conv = conv.transpose([0, 2, 3, 1])

    plt.figure()
    for i in range(64):
        plt.subplot(8, 8, i + 1)
        data = conv[i, :, :, :]
        data = (data - data.min()) / (data.max() - data.min())
        plt.imshow(data)
        plt.axis("off")
    plt.imshow()
    plt.savefig("filters.png")


def visualize_layer(layer_type):
    layer_type = layer_type.lower()
    if layer_type not in ("first", "last"):
        assert ValueError("Invalid parameters")

    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    dataset = datasets.CIFAR10(root='./data', train=False, download=False,
                               transform=transform)
    device = torch.device("cuda:0")
    image, label = dataset[random.randrange(len(dataset))]
    image = image.to(device, torch.float32).unsqueeze(0)

    if layer_type == 'first':
        net = ExtractFirstLayer(3)
        height, width = 8, 8  # 64
    elif layer_type == 'last':
        net = ExtractLastLayer(3)
        height, width = 16, 32  # 512
    else:
        assert "Error"
    net.load_state_dict(torch.load("./checkpoints/resnet/latest_net_resnet.pth"), strict=False)
    net.to(device)
    net.eval()

    with torch.no_grad():
        features, _ = net(image).cpu().numpy()
        image = image.cpu().numpy().squeeze().transpose([1, 2, 0])
        # print(image)
        image = (image + 1) / 2

    plt.figure(1)
    plt.imshow(image)
    plt.savefig("image_input_%s_layer.png" % layer_type)
    plt.axis("off")
    plt.figure(2)
    for i in range(height * width):
        plt.subplot(height, width, i + 1)
        data = features[0, i, :, :]
        plt.imshow(data, cmap="gray")
        plt.axis("off")
    plt.savefig("feature_map_%s_layer.png" % layer_type)


def vis_layer(layer):
    model = torch.load("./checkpoints/resnet/latest_net_resnet.pth")

    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    dataset = datasets.CIFAR10(root='./data', train=False, download=False,
                               transform=transform)
    device = torch.device("cuda:0")
    loader = torch.utils.data.DataLoader(dataset, batch_size=9, shuffle=True, num_workers=2)
    for i, data in enumerate(loader):
        if i == 0:
            image, label = data
            image = image.to(device, torch.float32)
            print(image.shape)
            break

    if layer == 1:
        net = ExtractFirstLayer(3)
        deconv_net = Deconv1_resnet()
        net.load_state_dict(torch.load("./checkpoints/resnet/latest_net_resnet.pth"), strict=False)
        deconv_net.deconv1.weight.data = model["conv.0.weight"]
    elif layer == 5:
        net = ExtractLastLayer(3)
        deconv_net = Deconv5_resnet()
        net.load_state_dict(torch.load("./checkpoints/resnet/latest_net_resnet.pth"), strict=False)
        deconv_net.deconv[0].shortcut.weight = net.conv[7].shortcut.weight
        deconv_net.deconv[0].conv[0].weight = net.conv[7].conv[3].weight
        deconv_net.deconv[0].conv[2].weight = net.conv[7].conv[0].weight
        deconv_net.deconv[1].shortcut.weight = net.conv[6].shortcut.weight
        deconv_net.deconv[1].conv[0].weight = net.conv[6].conv[3].weight
        deconv_net.deconv[1].conv[2].weight = net.conv[6].conv[0].weight
        deconv_net.deconv[2].shortcut.weight = net.conv[5].shortcut.weight
        deconv_net.deconv[2].conv[0].weight = net.conv[5].conv[3].weight
        deconv_net.deconv[2].conv[2].weight = net.conv[5].conv[0].weight
        deconv_net.deconv[3].shortcut.weight = net.conv[4].shortcut.weight
        deconv_net.deconv[3].conv[0].weight = net.conv[4].conv[3].weight
        deconv_net.deconv[3].conv[2].weight = net.conv[4].conv[0].weight
        deconv_net.deconv_last[1].weight = net.conv[0].weight

    net.to(device)
    net.eval()
    deconv_net.to(device)
    deconv_net.eval()

    recon_feature = [[]]

    with torch.no_grad():
        if layer == 1:
            num = 64
            features = net(image)
        elif layer == 5:
            num = 512
            features, indices = net(image)

        for i in range(9):
            plt.figure(i)
            for j in range(9):
                feature_copy = features.clone()
                if j == 0:
                    feature_copy[j + 1:, :, :, :] = 0
                elif j == 8:
                    feature_copy[:j, :, :, :] = 0
                else:
                    feature_copy[:j, :, :, :] = 0
                    feature_copy[j + 1:, :, :, :] = 0
                if i == 0:
                    feature_copy[:, 1:, :, :] = 0
                elif i == num - 1:
                    feature_copy[:, :i, :, :] = 0
                else:
                    feature_copy[:, :i, :, :] = 0
                    feature_copy[:, i + 1:, :, :] = 0
                if layer==5:
                    decon_output = deconv_net(feature_copy, indices)[j]
                else:
                    decon_output = deconv_net(feature_copy)[j]
                new_img = decon_output.cpu().numpy().squeeze().transpose([1, 2, 0])
                new_img = (new_img - new_img.min()) / (new_img.max() - new_img.min())
                plt.subplot(3, 3, j + 1)
                plt.imshow(new_img)
                plt.axis("off")
            plt.savefig("./image/layer%d_recon_feature_%d.png" % (layer, i))

        image = image.cpu().numpy()
        plt.figure()
        for i in range(9):
            img = image[i].copy()
            img = img.squeeze().transpose([1, 2, 0])
            img = (img + 1) / 2
            plt.subplot(3, 3, i + 1)
            plt.imshow(img)
            plt.axis("off")
        plt.savefig("./image/origin_image_%d.png" % layer)


if __name__ == "__main__":
    # vis_layer(5)
    vis_layer(1)
    # visualize_conv1_filter()
    # visualize_layer("first")
    # visualize_layer("last")
