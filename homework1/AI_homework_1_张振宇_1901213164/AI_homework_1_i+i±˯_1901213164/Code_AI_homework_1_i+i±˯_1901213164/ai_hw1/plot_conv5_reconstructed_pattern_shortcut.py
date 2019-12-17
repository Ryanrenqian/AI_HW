import torch
import  ZZYResNet18
model = ZZYResNet18.ZZYResNet18_indices(n_classes=10)
model.load_state_dict(torch.load("./cifar_net89.pth"))
model.eval()
conv1 = model.conv1
maxpool = model.maxpool
conv2_x = model.conv2_x
conv2_x_bb1_conv1 = conv2_x[0].conv1
conv2_x_bb1_conv2 = conv2_x[0].conv2
conv2_x_bb2_conv1 = conv2_x[1].conv1
conv2_x_bb2_conv2 = conv2_x[1].conv2


conv3_x = model.conv3_x
conv3_x_bb1_down_conv = conv3_x[0].down_conv
conv3_x_bb2_conv1 = conv3_x[1].conv1
conv3_x_bb2_conv2 = conv3_x[1].conv2


conv4_x = model.conv4_x
conv4_x_bb1_down_conv = conv4_x[0].down_conv
conv4_x_bb2_conv1 = conv4_x[1].conv1
conv4_x_bb2_conv2 = conv4_x[1].conv2


conv5_x = model.conv5_x
conv5_x_bb1_down_conv = conv5_x[0].down_conv
conv5_x_bb2_conv1 = conv5_x[1].conv1
conv5_x_bb2_conv2 = conv5_x[1].conv2


def deconv(feature_map, conv):
    output_padding = 0 if conv.stride[0] == 1 else 1
    return torch.nn.functional.conv_transpose2d(feature_map[:, :, :, :].transpose(2, 3), conv.weight[:, :, :, :],
                                                   bias=None, stride=conv.stride, padding=conv.padding, output_padding=output_padding, groups=1,
                                                   dilation=1)

class FeatureMapSaver():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.feature_map = output
    def close(self):
        self.hook.remove()

fms_conv5_x_bb2_conv2 = FeatureMapSaver(conv5_x_bb2_conv2)
class IndicesSaver():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.indices = output[1]
    def close(self):
        self.hook.remove()
is_maxpool = IndicesSaver(maxpool)
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
    feature_map_target = fms_conv5_x_bb2_conv2.feature_map  # ('b', 'c', 'h', 'w') # 1, 64, 112, 112

    #torch.relu()


    feature_map = deconv(feature_map_target, conv5_x_bb2_conv2)
    feature_map = torch.relu(feature_map)
    feature_map = deconv(feature_map, conv5_x_bb2_conv1)
    feature_map = torch.relu(feature_map)
    feature_map = deconv(feature_map, conv5_x_bb1_down_conv)
    feature_map = torch.relu(feature_map)


    feature_map = deconv(feature_map, conv4_x_bb2_conv2)
    feature_map = torch.relu(feature_map)
    feature_map = deconv(feature_map, conv4_x_bb2_conv1)
    feature_map = torch.relu(feature_map)
    feature_map = deconv(feature_map, conv4_x_bb1_down_conv)
    feature_map = torch.relu(feature_map)

    feature_map = deconv(feature_map, conv3_x_bb2_conv2)
    feature_map = torch.relu(feature_map)
    feature_map = deconv(feature_map, conv3_x_bb2_conv1)
    feature_map = torch.relu(feature_map)
    feature_map = deconv(feature_map, conv3_x_bb1_down_conv)
    feature_map = torch.relu(feature_map)

    feature_map = deconv(feature_map, conv2_x_bb2_conv2)
    feature_map = torch.relu(feature_map)
    feature_map = deconv(feature_map, conv2_x_bb2_conv1)
    feature_map = torch.relu(feature_map)
    feature_map = deconv(feature_map, conv2_x_bb1_conv2)
    feature_map = torch.relu(feature_map)
    feature_map = deconv(feature_map, conv2_x_bb1_conv1)
    feature_map = torch.relu(feature_map)

    feature_map = torch.nn.functional.max_unpool2d(feature_map, is_maxpool.indices, 3, stride=2, padding=0, output_size=(112, 112))
    feature_map = torch.relu(feature_map)
    img_tensor_in = deconv(feature_map, conv1)
    import cv2

    #pattern_img = torch.sigmoid(img_tensor_in[0, :, :, :].transpose(0, 2)).detach().numpy()
    pattern_img = img_tensor_in[0, :, :, :].transpose(0, 2).clamp(0, 1).detach()

    for i in range(pattern_img.shape[2]):
        pattern_img[:,:,i] = pattern_img[:,:,i] / pattern_img[:,:,i].max()
    # pattern_img = pattern_img / pattern_img.max()
    # pattern_img = pattern_img[:,:,:]
    pattern_img = zzytools.denormalize(pattern_img)
    for i in range(pattern_img.shape[2]):
        pattern_img[:,:,i] = pattern_img[:,:,i] / pattern_img[:,:,i].max()
    # pattern_img = pattern_img / pattern_img.max()
    return pattern_img


patterns_x16 = torch.cat([get_reconstructed_image(i).unsqueeze(0) for i in range(4531, 4531 + 16)])


zzytools.plot_rgb_images(patterns_x16.permute(1, 2, 3, 0).detach().numpy(), 'conv5_reconstruted_pattern', plot_dir='./conv5_reconstruted_pattern_shortcut')