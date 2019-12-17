import torch
import network
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import pickle
import matplotlib.pyplot as plt

model = network.resnet18()
model.load_state_dict(torch.load(r'./cifar_net_224.pth'))
model.eval()


class myNet(nn.Module):
    def _init_(self, pretrained_model, layers):
        super(self, myNet).__init__()
        self.visu_conv1 = nn.Sequential(*list(pretrained_model.conv1))

    def forward(self, x):
        out1 = self.visu_conv1(x)
        return out1


def get_features(pretrained_model, x):
    net1 = pretrained_model.conv1
    bn1 = pretrained_model.bn1
    relu = pretrained_model.relu
    maxpool = pretrained_model.maxpool
    conv2_x = pretrained_model.conv2_x
    conv3_x = pretrained_model.conv3_x
    conv4_x = pretrained_model.conv4_x
    conv5_x = pretrained_model.conv5_x

    out1 = net1(x)
    temp = bn1(out1)
    temp = relu(temp)
    temp = maxpool(temp)
    temp = conv2_x(temp)
    temp = conv3_x(temp)
    temp = conv4_x(temp)
    out5 = conv5_x(temp)
    return out1, out5


transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor()
    ]
)
set = torchvision.datasets.CIFAR10(root=r'./', train=True,
                                   download=True, transform=transform)
loader = torch.utils.data.DataLoader(set, batch_size=1,
                                     shuffle=True, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

for i, data in enumerate(loader):
    x = data[0].to(device)
    print("label:", data[1])
    # print(x.size())  # torch.Size([1, 3, 32, 32])
    out1, out5 = get_features(model, x)
    out1 = out1.cpu().clone()  # cuda tensor转cpu tensor
    out5 = out5.cpu().clone()
    print(out1.size())  # torch.Size([1, 64, 16, 16])
    print(out5.size())  # torch.Size([1, 512, 1, 1])
    out1 = out1.squeeze(0)
    out5 = out5.squeeze(0)

    f, ax = plt.subplots(8, 8)
    for i in range(8):
        for j in range(8):
            feature = out1[8 * i + j]
            feature = feature.detach().numpy()
            ax[i][j].imshow(feature)
            ax[i][j].axis("off")

    # feature = out1[0]
    # # print(feature.size())  # torch.Size([16, 16])
    # feature = feature.detach().numpy()
    # plt.imshow(feature)
    plt.show()

    f, ax2 = plt.subplots(16, 32)
    for m in range(16):
        for n in range(32):
            feature2 = out5[16 * m + n]
            feature2 = feature2.detach().numpy()
            ax2[m][n].imshow(feature2)
            ax2[m][n].axis("off")
    plt.show()

    x = x.cpu().clone()
    x = x.squeeze(0)
    # print(x.size())  # torch.Size([3, 32, 32])
    unloader = transforms.ToPILImage()
    x = unloader(x)
    # print(x)
    x.show()

    break

# to_pil_image = transforms.ToPILImage()
# for image, label in loader:
#     print("label:", label)
#     img = to_pil_image(image[0])
#     img.show()
#     break

# print(out1.size())
