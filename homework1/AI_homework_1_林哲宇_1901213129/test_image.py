import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam
from torch.autograd import Variable
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
import MyResnet

# Datasets CIFAR10

transform = transforms.Compose([transforms.Resize((224,224)), \
    transforms.ToTensor(), \
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Get image

dataiter = iter(testloader)
images, labels = dataiter.next()

# Load Model

device = "cpu"
net = MyResnet.ResNet(MyResnet.ResBlock).to(device)

PATH = './cifar_net_224.pth'
net.load_state_dict(torch.load(PATH,map_location='cpu'))

# outputs for activation
outputs, conv1_output, conv5_output = net(images)
print(conv1_output.size())

# compute activatioin
loss = torch.mean(conv1_output[0,1])
print(loss)
print(loss.data.cpu().numpy())

#column graph conv1_output
x_filters_conv1 = []
y_activations_conv1 = []
for i, feature in enumerate(conv1_output[0]):
	x_filters_conv1.append(i)
	y_activations_conv1.append(torch.mean(feature).data.cpu().numpy())
plt.bar(x=x_filters_conv1, height=y_activations_conv1, color='steelblue')
plt.show()

#column graph conv5_output
x_filters_con5 = []
y_activations_con5 = []
for i, feature in enumerate(conv5_output[0]):
	x_filters_con5.append(i)
	y_activations_con5.append(torch.mean(feature).data.cpu().numpy())
plt.bar(x=x_filters_con5, height=y_activations_con5, color='steelblue')
plt.show()

