import resnet_deconv
from torchvision import transforms
import torchvision
import torch
import network
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image


def get_features(pretrained_model, x):
    conv1 = pretrained_model.conv1
    bn1 = pretrained_model.bn1
    relu = pretrained_model.relu
    maxpool = pretrained_model.maxpool
    conv2_x = pretrained_model.conv2_x
    conv3_x = pretrained_model.conv3_x
    conv4_x = pretrained_model.conv4_x
    conv5_x = pretrained_model.conv5_x

    out1 = conv1(x)
    temp = bn1(out1)
    temp = relu(temp)
    temp, indices = maxpool(temp)
    temp = conv2_x(temp)
    temp = conv3_x(temp)
    temp = conv4_x(temp)
    out5 = conv5_x(temp)
    return out5, indices, out1


def getdeconv(deconv_model, x, indices):
    deconv5_x = deconv_model.conv5_x
    deconv4_x = deconv_model.conv4_x
    deconv3_x = deconv_model.conv3_x
    deconv2_x = deconv_model.conv2_x
    maxunpool = deconv_model.maxunpool
    relu = deconv_model.relu
    bn1 = deconv_model.bn1
    deconv1 = deconv_model.conv1

    deconv_5_x = deconv5_x(x)
    temp = deconv4_x(deconv_5_x)
    temp = deconv3_x(temp)
    temp = deconv2_x(temp)
    temp = maxunpool(temp, indices)
    temp = relu(temp)
    temp = bn1(temp)
    deconv_1 = deconv1(temp)
    return deconv_5_x, deconv_1


resnet18_without_fc = resnet_deconv.resnet18_without_fc()
deconv_resnet18_without_fc = resnet_deconv.deconv_resnet18_without_fc()
deconv_layer1 = resnet_deconv.deconv_layer1()
# print(resnet18_without_fc)
# print(deconv_resnet18_without_fc)

transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor()
    ]
)
set = torchvision.datasets.CIFAR10(root=r'D:\研一\人工智能\Assigment1', train=True,
                                   download=True, transform=transform)
loader = torch.utils.data.DataLoader(set, batch_size=1,
                                     shuffle=True, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet18_without_fc.to(device)
deconv_resnet18_without_fc.to(device)

model = network.resnet18()
model.load_state_dict(torch.load(r'D:\研一\人工智能\Assigment1\cifar_net_224.pth'))

resnet18_without_fc_dict = resnet18_without_fc.state_dict()

model_dict = model.state_dict()
model_dict = {k: v for k, v in model_dict.items() if k in resnet18_without_fc_dict}
resnet18_without_fc_dict.update(model_dict)
resnet18_without_fc.load_state_dict(model_dict)

# for k, v in resnet18_without_fc.state_dict().items():
#     print(k)
#     # print(v.size())
# print("--------------------------------------")
# for k, v in deconv_resnet18_without_fc.state_dict().items():
#     print(k)
#     # print(v.size())

deconv_resnet18_without_fc.conv5_x[1].conv1.weight = resnet18_without_fc.conv5_x[0].conv1.weight
deconv_resnet18_without_fc.conv5_x[1].bn1 = resnet18_without_fc.conv5_x[0].bn1
deconv_resnet18_without_fc.conv5_x[1].conv2.weight = resnet18_without_fc.conv5_x[0].conv2.weight
deconv_resnet18_without_fc.conv5_x[1].bn2 = resnet18_without_fc.conv5_x[0].bn2
deconv_resnet18_without_fc.conv5_x[0].conv1.weight = resnet18_without_fc.conv5_x[1].conv1.weight
deconv_resnet18_without_fc.conv5_x[0].bn1 = resnet18_without_fc.conv5_x[1].bn1
deconv_resnet18_without_fc.conv5_x[0].conv2.weight = resnet18_without_fc.conv5_x[1].conv2.weight
deconv_resnet18_without_fc.conv5_x[0].bn2 = resnet18_without_fc.conv5_x[1].bn1

deconv_resnet18_without_fc.conv4_x[1].conv1.weight = resnet18_without_fc.conv4_x[0].conv1.weight
deconv_resnet18_without_fc.conv4_x[1].bn1 = resnet18_without_fc.conv4_x[0].bn1
deconv_resnet18_without_fc.conv4_x[1].conv2.weight = resnet18_without_fc.conv4_x[0].conv2.weight
deconv_resnet18_without_fc.conv4_x[1].bn2 = resnet18_without_fc.conv4_x[0].bn2
deconv_resnet18_without_fc.conv4_x[0].conv1.weight = resnet18_without_fc.conv4_x[1].conv1.weight
deconv_resnet18_without_fc.conv4_x[0].bn1 = resnet18_without_fc.conv4_x[1].bn1
deconv_resnet18_without_fc.conv4_x[0].conv2.weight = resnet18_without_fc.conv4_x[1].conv2.weight
deconv_resnet18_without_fc.conv4_x[0].bn2 = resnet18_without_fc.conv4_x[1].bn1

deconv_resnet18_without_fc.conv3_x[1].conv1.weight = resnet18_without_fc.conv3_x[0].conv1.weight
deconv_resnet18_without_fc.conv3_x[1].bn1 = resnet18_without_fc.conv3_x[0].bn1
deconv_resnet18_without_fc.conv3_x[1].conv2.weight = resnet18_without_fc.conv3_x[0].conv2.weight
deconv_resnet18_without_fc.conv3_x[1].bn2 = resnet18_without_fc.conv3_x[0].bn2
deconv_resnet18_without_fc.conv3_x[0].conv1.weight = resnet18_without_fc.conv3_x[1].conv1.weight
deconv_resnet18_without_fc.conv3_x[0].bn1 = resnet18_without_fc.conv3_x[1].bn1
deconv_resnet18_without_fc.conv3_x[0].conv2.weight = resnet18_without_fc.conv3_x[1].conv2.weight
deconv_resnet18_without_fc.conv3_x[0].bn2 = resnet18_without_fc.conv3_x[1].bn1

deconv_resnet18_without_fc.conv2_x[1].conv1.weight = resnet18_without_fc.conv2_x[0].conv1.weight
deconv_resnet18_without_fc.conv2_x[1].bn1 = resnet18_without_fc.conv2_x[0].bn1
deconv_resnet18_without_fc.conv2_x[1].conv2.weight = resnet18_without_fc.conv2_x[0].conv2.weight
deconv_resnet18_without_fc.conv2_x[1].bn2 = resnet18_without_fc.conv2_x[0].bn2
deconv_resnet18_without_fc.conv2_x[0].conv1.weight = resnet18_without_fc.conv2_x[1].conv1.weight
deconv_resnet18_without_fc.conv2_x[0].bn1 = resnet18_without_fc.conv2_x[1].bn1
deconv_resnet18_without_fc.conv2_x[0].conv2.weight = resnet18_without_fc.conv2_x[1].conv2.weight
deconv_resnet18_without_fc.conv2_x[0].bn2 = resnet18_without_fc.conv2_x[1].bn1

deconv_resnet18_without_fc.bn1 = resnet18_without_fc.bn1
deconv_resnet18_without_fc.conv1.weight = resnet18_without_fc.conv1.weight

deconv_layer1.deconv1.weight = resnet18_without_fc.conv1.weight

for i, data in enumerate(loader):
    x = data[0].to(device)
    # print("label:", data[1])
    # print("x size:", x.size())  # torch.Size([1, 3, 224, 224])
    out5, indices, out1 = get_features(resnet18_without_fc, x)
    # print(out5)
    # out5 = out5.cpu().clone()
    # print("conv5_x output size:", out5.size())  # torch.Size([1, 512, 7, 7])
    # out5 = out5.squeeze(0)
    deconv_5_x, deconv_1 = getdeconv(deconv_resnet18_without_fc, out5, indices)
    deconv_5_x = deconv_5_x.cpu().clone()
    deconv_5_x = deconv_5_x.squeeze(0)

    deconv_1 = deconv_1.cpu().clone()
    deconv_1 = deconv_1.squeeze(0)
    # print("deconv_5_x size:", deconv_5_x.size())
    # print("deconv_1 size:", deconv_1.size())

    # f, ax = plt.subplots(16, 16)
    # for i in range(16):
    #     for j in range(16):
    #         feature = deconv_5_x[16 * i + j]
    #         feature = feature.detach().numpy()
    #         ax[i][j].imshow(feature)
    #         ax[i][j].axis("off")
    # plt.show()

    unloader = transforms.ToPILImage()
    deconv1 = deconv_layer1(out1)
    deconv1 = deconv1.cpu().clone()
    # print(deconv1.size())  # torch.Size([1, 3, 223, 223])
    deconv1 = deconv1.squeeze(0)
    deconv1 = unloader(deconv1)
    deconv1.show()
    deconv_1 = unloader(deconv_1)
    deconv_1.show()
    #
    x = x.cpu().clone()
    x = x.squeeze(0)
    # print(x.size())  # torch.Size([3, 224, 224])

    x = unloader(x)
    x.show()


    break
