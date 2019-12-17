import torch
import network
import matplotlib.pyplot as plt
from torchsummary import summary

model = network.resnet18()
model.load_state_dict(torch.load(r'./cifar_net_224.pth'))
# model2 = network.resnet18()
# model2.conv1.weight = model.conv1.weight
# model2_dict = model.state_dict()
# print(model2_dict)
# print(model)
# print(model.conv1.weight[0].shape)  # [3,7,7]
# print(model.conv1.weight[0])

# 取出conv1的权重矩阵
conv1_weight_numpy = model.conv1.weight.detach().numpy()
# print(conv1_weight_numpy.shape)  # (64, 3, 7, 7)

# conv5_x_weight_numpy = model.conv5_x[0].bn1.weight
# print(conv5_x_weight_numpy)

# 权重矩阵取第一个通道
conv1_weight_numpy_ave = conv1_weight_numpy[:, 0, :, :]
# print(conv1_weight_numpy_ave.shape)

# 一共64个filter，通过8x8子图可视化
f, ax = plt.subplots(8, 8)
for i in range(8):
    for j in range(8):
        ax[i][j].matshow(conv1_weight_numpy_ave[8 * i + j])
        ax[i][j].axis("off")  # 关闭子图标签

plt.show()

