import torch
from torch import nn
from misc_functions import preprocess_image, recreate_image, save_image
import os
from torch.autograd import Variable
from torch.optim import Adam

class Backprop():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """

    def __init__(self, model, selected_layer, selected_filter, pic, lr, png_dir):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.pic = pic
        self.png_dir = png_dir
        self.selected_filter = selected_filter
        self.conv_output = 0
        self.lr = lr
        # Create the folder to export images if not exists
        if not os.path.exists('./generated/' + self.png_dir):
            os.makedirs('./generated/' + self.png_dir)

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]

        # Hook the selected layer
        self.model[self.selected_layer].register_forward_hook(hook_function)

    def visualise_layer_without_hooks(self):
        # Process image and return variable
        processed_image = self.pic[None, :, :, :]
        processed_image = processed_image.cuda()
        processed_image = Variable(processed_image, requires_grad=True)
        # Define optimizer for the image
        optimizer = Adam([processed_image], lr=self.lr, weight_decay=1e-6)
        for i in range(1, 201):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = processed_image
            for name, module in self.model._modules.items():
                # Forward pass layer by layer
                x = module(x)
                if name == self.selected_layer:
                    # Only need to forward until the selected layer is reached
                    # Now, x is the output of the selected layer
                    break
            self.conv_output = x[0, self.selected_filter]
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.cpu().numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Save image
            if i % 200 == 0:
                # Recreate image
                processed_image = processed_image.cpu()
                self.created_image = recreate_image(processed_image)
                im_path = './generated/' + self.png_dir + '/layer_vis_' + str(self.selected_layer) + \
                          '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                save_image(self.created_image, im_path)