import torch
import  ZZYResNet18
model = ZZYResNet18.ZZYResNet18(n_classes=10)
model.load_state_dict(torch.load("./cifar_net89.pth"))
model.eval()
conv1 = model.conv1

class FeatureMapSaver():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.feature_map = output
    def close(self):
        self.hook.remove()

fms_conv1 = FeatureMapSaver(conv1)

import torch
import torchvision
import torchvision.transforms as transforms

resize_to_tensor = transforms.Compose(
    [transforms.Resize((224, 224)),
    transforms.ToTensor()])
norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                           download=True, transform=resize_to_tensor)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=0)
import zzytools

def get_reconstructed_image(ith):
    input_img, label = testloader.dataset[ith]
    input_img = norm(input_img)
    print(label)
    model(input_img.unsqueeze(0))
    feature_map = fms_conv1.feature_map #  ('b', 'c', 'h', 'w') # 1, 64, 112, 112
    w_tensor = model.conv1.weight #  'out', 'in', 'h', 'w')# 64 , 3, 7, 7


    import matplotlib.pyplot as plt
    import cv2
    pattern = torch.nn.functional.conv_transpose2d(feature_map[:, 0:64, :, :].transpose(2, 3), w_tensor[0:64,:,:,:], bias=None, stride=2, padding=3, output_padding=1, groups=1, dilation=1)
    pattern = pattern[0, :, :, :].transpose(0, 2).clamp(0, 1).detach()
    #pattern = zzytools.denormalize(pattern)
    return pattern
    #pattern_img = torch.sigmoid(pattern[0, :, :, :].transpose(0, 2)).detach().numpy()
    #pattern_img = pattern_img[:,:,:]
    #pattern_img_bgr = cv2.cvtColor(pattern_img, cv2.COLOR_RGB2BGR)
    #cv2.imshow('aaa', pattern_img_bgr)
    #cv2.waitKey(0)

patterns_x16 = torch.cat([get_reconstructed_image(i).unsqueeze(0) for i in range(4531, 4531 + 16)])


zzytools.plot_rgb_images(patterns_x16.permute(1, 2, 3, 0).detach().numpy(), 'conv1_reconstruted_pattern', plot_dir='./conv1_reconstruted_pattern')

def get_ori_image(ith):
    input_img, label = testloader.dataset[ith]
    #input_img=zzytools.denormalize_transpose(norm(input_img))
    return input_img
img_x16 = torch.cat([get_ori_image(i).unsqueeze(0) for i in range(4531, 4531 + 16)])
import zzytools

zzytools.plot_rgb_images(img_x16.permute(2, 3, 1, 0).detach().numpy(), 'ori_image', plot_dir='./conv1_reconstruted_pattern')