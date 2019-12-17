"""
Created on Sat Nov 18 23:12:08 2017
@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import numpy as np

import torch
from torch.optim import Adam
from torchvision import models

from misc_functions import preprocess_image, recreate_image, save_image


class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        # Create the folder to export images if not exists
        if not os.path.exists('./generated'):
            os.makedirs('./generated')

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]
        # Hook the selected layer
        self.model[self.selected_layer].register_forward_hook(hook_function)

    def visualise_layer_with_hooks(self):
        # Hook the selected layer
        self.hook_layer()
        # Generate a random image
        random_image = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))
        # Process image and return variable
        processed_image = preprocess_image(random_image, False)
        # Define optimizer for the image
        optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
        for i in range(1, 31):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = processed_image
            for index, layer in enumerate(self.model):
                # Forward pass layer by layer
                # x is not used after this point because it is only needed to trigger
                # the forward hook function
                x = layer(x)
                # Only need to forward until the selected layer is reached
                if index == self.selected_layer:
                    # (forward hook function triggered)
                    break
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(processed_image)
            # Save image
            if i % 30 == 0:
                im_path = './generated/layer_vis_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                save_image(self.created_image, im_path)

    def visualise_layer_without_hooks(self, images):
        # Process image and return variable
        # Generate a random image
        # random_image = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))
        from PIL import Image
        from torch.autograd import Variable
        # img_path = '/home/hzq/tmp/waibao/data/cat_dog.png'

        processed_image = images
        processed_image = Variable(processed_image, requires_grad=True)

        # Define optimizer for the image
        optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
        for i in range(1, 31):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = processed_image
            # for index, layer in enumerate(self.model):
            #     # Forward pass layer by layer
            #     x = layer(x)
            #     if index == self.selected_layer:
            #         # Only need to forward until the selected layer is reached
            #         # Now, x is the output of the selected layer
            #         break
            for name, module in self.model._modules.items():
                # Forward pass layer by layer
                x = module(x)
                print(name)
                if name == self.selected_layer:
                    # Only need to forward until the selected layer is reached
                    # Now, x is the output of the selected layer
                    break



            # Here, we get the specific filter from the output of the convolution operation
            # x is a tensor of shape 1x512x28x28.(For layer 17)
            # So there are 512 unique filter outputs
            # Following line selects a filter from 512 filters so self.conv_output will become
            # a tensor of shape 28x28
            self.conv_output = x[0, self.selected_filter]

            # feature map



            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(processed_image)
            # Save image
            if i % 30 == 0:
                im_path = './generated/layer_vis_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                save_image(self.created_image, im_path)

from matplotlib import pyplot as plt

def layer_output_visualization(model, selected_layer, selected_filter, pic, png_dir, iter):
    pic = pic[None,:,:,:]
    # pic = pic.cuda()
    x = pic
    x = x.squeeze(0)
    for name, module in model._modules.items():
        x = module(x)
        if name == 'layer1':
            import torch.nn.functional as F
            x = F.max_pool2d(x, 3, stride=2, padding=1)
        if name == selected_layer:
            break
    conv_output = x[0, selected_filter]
    x = conv_output.cpu().detach().numpy()
    # print(x.shape)
    if not os.path.exists('./output/'+ png_dir):
        os.makedirs('./output/'+ png_dir)
    im_path = './output/'+ png_dir+ '/layer_vis_' + str(selected_layer) + \
                    '_f' + str(selected_filter) + '_iter' + str(iter) + '.jpg'
    plt.imshow(x, cmap = plt.cm.jet)
    plt.axis('off')
    plt.savefig(im_path)

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) * 255 / _range

def filter_visualization(model, selected_layer, selected_filter, png_dir):
    for name, param in model.named_parameters():
        print(name)
        if name == selected_layer + '.weight':
            x = param
    x = x[selected_filter,:,:,:]
    x = x.cpu().detach().numpy()
    x = x.transpose(1,2,0)
    x = normalization(x)
    x = preprocess_image(x, resize_im=False)
    x = recreate_image(x)
    if not os.path.exists('./filter/'+ png_dir):
        os.makedirs('./filter/'+ png_dir)
    im_path = './filter/'+ png_dir+ '/layer_f' + str(selected_filter) + '_iter' + '.jpg'
    # save_image(x, im_path)
    plt.imshow(x, cmap = plt.cm.jet)
    plt.axis('off')
    plt.savefig(im_path)



if __name__ == '__main__':
    cnn_layer = 'conv1'
    # cnn_layer = 'layer4'
    filter_pos = 5
    # Fully connected layer is not needed

    # use resnet insted
    from cifar import ResNet18, ResidualBlock
    pretrained_model = ResNet18(ResidualBlock)
    pretrained_model.load_state_dict(torch.load("models/net_044.pth"))

    print(
        pretrained_model
    )
    # pretrained_model = models.vgg16(pretrained=True).features

    import torchvision

    def transform(x):
        x = x.resize((224, 224), 2)  # resize the image from 32*32 to 224*224
        x = np.array(x, dtype='float32') / 255
        x = (x - 0.5) / 0.5  # Normalize
        x = x.transpose((2, 0, 1))  # reshape, put the channel to 1-d;  input = {channel, size, size}
        x = torch.from_numpy(x)
        return x


    trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=2)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    images, labels = iter(trainloader).next()
    images, labels = iter(trainloader).next()
    images, labels = iter(trainloader).next()
    images, labels = iter(trainloader).next()
    images, labels = iter(trainloader).next()
    # images, labels = iter(trainloader).next()
    # images, labels = iter(trainloader).next()

    x = images.cpu()[0, :, :, :]
    x = x[None,:,:,:]
    x = recreate_image(x)
    save_image(x, 'orginal.jpg')



    # images, labels = images.to(device), labels.to(device)  # 有

    # for filter_pos in range(64):
    #
    #     layer_vis = CNNLayerVisualization(pretrained_model, cnn_layer, filter_pos)
    #     layer_vis.visualise_layer_without_hooks(images)

        # Layer visualization with pytorch hooks
        # layer_vis.visualise_layer_with_hooks()

        # Layer visualization without pytorch hooks

    for filter_pos in range(64):
        # # get the filter
        filter_visualization(model = pretrained_model, selected_layer = 'conv1.0', selected_filter = filter_pos, png_dir = 'conv1.0')

        # get the feature map : conv1
        layer_output_visualization(model = pretrained_model, selected_layer = 'conv1', selected_filter = filter_pos, pic = images, png_dir = 'conv1', iter = filter_pos)

        # get the feature map: layer4
        layer_output_visualization(model = pretrained_model, selected_layer = 'layer4', selected_filter = filter_pos, pic = images, png_dir = 'layer4', iter = filter_pos)

        # get the reconstruct
        layer_vis = CNNLayerVisualization(pretrained_model, cnn_layer, filter_pos)
        layer_vis.visualise_layer_without_hooks(images)


    for filter_pos in range(512):
        layer_output_visualization(model = pretrained_model, selected_layer = 'layer4', selected_filter = filter_pos, pic = images, png_dir = 'layer4', iter = filter_pos)

        layer_vis = CNNLayerVisualization(pretrained_model, cnn_layer, filter_pos)
        layer_vis.visualise_layer_without_hooks(images)

