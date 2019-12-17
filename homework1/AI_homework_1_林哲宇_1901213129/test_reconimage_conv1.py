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

import cv2
from cv2 import resize
from PIL import Image,ImageOps
import copy

# load model
device = "cpu"
net = MyResnet.ResNet(MyResnet.ResBlock).to(device)

PATH = './cifar_net_224.pth'
net.load_state_dict(torch.load(PATH,map_location='cpu'))

# Generate a random image
random_image = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))
# print(random_image)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

random_image = transform(random_image)
# print(random_image)

random_image = random_image.unsqueeze_(0)
# print(random_image.size())

processed_image = Variable(random_image, requires_grad=True)
# print(processed_image)

# Define optimizer for the image
optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)

for i in range(1, 31):
	optimizer.zero_grad()
	# Assign create image to a variable to move forward in the model
	x = processed_image
	outputs, conv1_output, conv5_output = net(processed_image)
	conv1_output_filter = conv1_output[0][43]
	# print(conv1_output_filter.size())
	loss = -torch.mean(conv1_output_filter)
	print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
	# Backward
	loss.backward()
	# Update image
	optimizer.step()

# Recreate image 1
# created_image = recreate_image(processed_image)
recreated_image = processed_image.cpu().clone()
recreated_image = recreated_image.squeeze(0)
unloader = transforms.ToPILImage()
recreated_image = unloader(recreated_image)
recreated_image = cv2.cvtColor(np.asarray(recreated_image),cv2.COLOR_RGB2BGR)
cv2.imwrite('res_conv1_43.jpg',recreated_image)