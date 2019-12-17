'''
zhuliwen: liwenzhu@pku.edu.cn
October 24，2019
ref: https://github.com/jacobgil/pytorch-grad-cam
'''

import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import utils
import cv2
import numpy as np
from AI_homework_1 import ResNet, BasicBlock
import os


class FeatureExtractor():

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model.conv1, target_layers)


    def get_gradients(self):
        return self.feature_extractor.gradients


    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = self.model.conv2_x(output)
        output = self.model.conv3_x(output)
        output = self.model.conv4_x(output)
        output = self.model.conv5_x(output)
        output = self.model.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.model.fc(output)
        return target_activations, output


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = Variable(preprocessed_img, requires_grad=True)
    return input


def show_cam_on_image(img, mask, image_path):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite('./img_fmap/'+ os.path.splitext(image_path)[0]+"_after_cam.jpg", np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        one_hot.backward()
        print(self.extractor.get_gradients())

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        # cam = cam / (np.max(cam) - np.min(cam))

        return cam


class GuidedBackpropReLU(Function):

    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        for idx, module in self.model.conv1._modules.items():
            if module.__class__.__name__ == 'ReLU':
                self.model.conv1._modules[idx] = GuidedBackpropReLU()

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        one_hot.backward()

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


if __name__ == '__main__':
    net = ResNet(BasicBlock, [2, 2, 2, 2])
    net = net.cuda()
    net.load_state_dict(torch.load('resnet18-25-best.pth'))

    for image_path in os.listdir('./img_fmap/'):
        if 'after' not in image_path:
            net = ResNet(BasicBlock, [2, 2, 2, 2])
            net = net.cuda()
            net.load_state_dict(torch.load('resnet18-25-best.pth'))
            print(image_path)
            img = cv2.imread('./img_fmap/'+ image_path, 1)
            img = np.float32(cv2.resize(img, (224, 224))) / 255
            input = preprocess_image(img)

            target_index = None
            grad_cam = GradCam(model=net, \
                               target_layer_names=["1"], use_cuda=True)

            mask = grad_cam(input, target_index)

            show_cam_on_image(img, mask, image_path)

            gb_model = GuidedBackpropReLUModel(model=net, use_cuda=True)
            gb = gb_model(input, index=target_index)
            utils.save_image(torch.from_numpy(gb), './img_fmap/'+ os.path.splitext(image_path)[0]+'_after_gb.jpg')

            cam_mask = np.zeros(gb.shape)
            for i in range(0, gb.shape[0]):
                cam_mask[i, :, :] = mask

            cam_gb = np.multiply(cam_mask, gb)
            utils.save_image(torch.from_numpy(cam_gb), './img_fmap/'+ os.path.splitext(image_path)[0]+'_after_cam_gb.jpg')
