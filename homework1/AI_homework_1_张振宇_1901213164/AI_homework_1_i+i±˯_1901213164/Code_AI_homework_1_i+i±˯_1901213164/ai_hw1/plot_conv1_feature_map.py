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

input_img, _ = testloader.dataset[7876]
input_img = norm(input_img)
model(input_img.unsqueeze(0))
fms_conv1.feature_map

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


#grid = torchvision.utils.make_grid(quantize(fms_conv1.feature_map.transpose(0, 1), 1))
#grid = torchvision.utils.make_grid(quantize(fms_conv1.feature_map.transpose(0, 1), 1))
#from torch.utils.tensorboard import SummaryWriter



#writer = SummaryWriter(log_dir='./runs/conv1_feature_map_on_7876', comment='')
#writer.add_image('conv1_featuremap', grid, 0)
#writer.close()

import zzytools

w_tensor = fms_conv1.feature_map.refine_names('b', 'c', 'h', 'w')



zzytools.plot_conv_feature_map(w_tensor.align_to('h', 'w', 'b', 'c').detach().numpy(), 'conv1_feature_map_7876', plot_dir='./conv1_feature_map')