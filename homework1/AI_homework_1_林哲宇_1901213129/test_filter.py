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
import math
import os

# load model

# device = "cpu"
# net = MyResnet.ResNet(MyResnet.ResBlock).to(device)

# PATH = './cifar_net_224.pth'
# net.load_state_dict(torch.load(PATH,map_location='cpu'))

# Parameters

Epoch = 25
Batch_size = 128
LR = 0.01 

# Datasets CIFAR10

transform = transforms.Compose([transforms.Resize((224,224)), \
    transforms.ToTensor(), \
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=Batch_size,
                                          shuffle=True, num_workers=2)

# Load Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = MyResnet.ResNet(MyResnet.ResBlock).to(device)

# Loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = LR, momentum = 0.9, weight_decay=5e-4)

# Train 

for epoch in range(Epoch):
    net.train()
    print("Training")
    running_loss = 0.0
    accuracy = 0.0
    counter = 0.0
    for i, data in enumerate(trainloader, 0):
        length = len(trainloader)
        # print(length)
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs, conv1_output, conv5_output = net(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        counter += labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        accuracy += (predicted == labels).sum().item()
        print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), running_loss / (i + 1), 100. * accuracy / counter))

print('Finished Training')

# output weights
PLOT_DIR = './out/plots'

def create_dir(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise

def prepare_dir(path, empty=False):
    if not os.path.exists(path):
        create_dir(path)

def prime_powers(n):
    factors = set()
    for x in range(1, int(math.sqrt(n)) + 1):
        if n % x == 0:
            factors.add(int(x))
            factors.add(int(n // x))
    return sorted(factors)
    
def get_grid_dim(x):
    factors = prime_powers(x)
    if len(factors) % 2 == 0:
        i = int(len(factors) / 2)
        return factors[i], factors[i - 1]

    i = len(factors) // 2
    return factors[i], factors[i]

def plot_conv_weights(weights, name, channels_all=True):

    plot_dir = os.path.join(PLOT_DIR, 'conv_weights')
    plot_dir = os.path.join(plot_dir, name)
    prepare_dir(plot_dir, empty=True)

    w_min = np.min(weights)
    w_max = np.max(weights)

    channels = [0]
    if channels_all:
        channels = range(weights.shape[2])

    num_filters = weights.shape[3]
    grid_r, grid_c = get_grid_dim(num_filters)
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))
    # iterate channels
    for channel in channels:
        for l, ax in enumerate(axes.flat):
            img = weights[:, :, channel, l]
            ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
            ax.set_xticks([])
            ax.set_yticks([])
        plt.savefig(os.path.join(plot_dir, '{}-{}.png'.format(name, channel)), bbox_inches='tight')


# Get weights
for param in net.parameters():
	print(param.size())
	conv1_para = param.data
	break

# Preproces weights
print(type(conv1_para))
conv1_para = conv1_para.permute(2,3,1,0)
conv1_para = conv1_para.detach().cpu().numpy()

# Show weights
plot_conv_weights(conv1_para, 'conv_weights{}'.format(0))