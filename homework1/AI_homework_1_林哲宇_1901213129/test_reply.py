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
device = "cpu"
net = MyResnet.ResNet(MyResnet.ResBlock).to(device)

PATH = './cifar_net_224.pth'
net.load_state_dict(torch.load(PATH,map_location='cpu'))

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

def plot_conv_output(conv_img, name):
	plot_dir = os.path.join(PLOT_DIR, 'conv_output')
	plot_dir = os.path.join(plot_dir, name)
	prepare_dir(plot_dir, empty=True)

	w_min = np.min(conv_img)
	w_max = np.max(conv_img)
	num_filters = conv_img.shape[3]
	grid_r, grid_c = get_grid_dim(num_filters)
	fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

	# iterate filters
	for l, ax in enumerate(axes.flat):
		img = conv_img[0, :, :,  l]
		ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='bicubic', cmap='Greys')
		ax.set_xticks([])
		ax.set_yticks([])
	plt.savefig(os.path.join(plot_dir, '{}.png'.format(name)), bbox_inches='tight')


# Get display image
dataiter = iter(testloader)
images, labels = dataiter.next()

# Get conv_outputs
outputs, conv1_output, conv5_output = net(images)

conv1_output = conv1_output.permute(0, 2, 3, 1)
conv1_output = conv1_output.detach().numpy()
plot_conv_output(conv1_output, 'conv{}'.format(1))

conv5_output = conv5_output.permute(0, 2, 3, 1)
conv5_output = conv5_output.detach().numpy()
plot_conv_output(conv5_output, 'conv{}'.format(5))
