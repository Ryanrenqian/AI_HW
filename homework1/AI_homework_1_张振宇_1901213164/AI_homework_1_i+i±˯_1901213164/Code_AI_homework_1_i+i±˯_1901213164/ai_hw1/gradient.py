import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
#%matplotlib inline
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = output
    def close(self):
        self.hook.remove()

transform = transforms.Compose(
    [transforms.Resize((224, 224)),
    transforms.ToTensor()])
norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=24)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=0)
model = models.resnet18(num_classes=10)
model.load_state_dict(torch.load("./cifar_net84.pth"))
model.eval()

layer = list(list(model.children())[7].children())[1].conv2
activations = SaveFeatures(layer)

model.to(device)

img_for_plot = None
# testloader.dataset[0]
# for i,(temp_img,label) in enumerate(testloader):
#     img_for_plot = temp_img.clone()
#     temp_img = norm(temp_img[0,:,:,:])
#     temp_img = temp_img.to(device)
#     model(temp_img.expand(1,-1,-1,-1))
max_index = 0
max_value = 0
max_activation_image = None
max_activation_input = None
for i in range(10000):
    temp_img, label = testloader.dataset[i]
    img_for_plot = temp_img.clone()
    temp_img = norm(temp_img)
    temp_img = temp_img.to(device)
    input = temp_img.expand(1, -1, -1, -1)
    model(input)
    print(activations.features[0, 0].sum())
    v = activations.features[0, 0].sum()
    if i == 0:
        max_value = v
    else:
        if v > max_value:
            max_index = i
            max_value = v
            max_activation_image = img_for_plot.clone()
            max_activation_input = input.clone()

import numpy as np
img = np.uint8(np.random.uniform(150, 180, (1, 3, 224, 224)))/255  # generate random image
img.shape

print(max_index)

img = np.uint8(np.random.uniform(150, 180, (1, 3, 224, 224))) / 255
img_tensor = torch.from_numpy(img)
img_tensor = img_tensor.float().to(device)
img_tensor.requires_grad = True
print(img_tensor.shape)
model.to(device)
# layer = list(list(model.children())[7].children())[1].conv2
optimizer = torch.optim.Adam([img_tensor], lr=0.001, weight_decay=1e-6)
# activations = SaveFeatures(layer)
mse = torch.nn.MSELoss()
import time
for n in range(200):
    optimizer.zero_grad()
    model(img_tensor)
    loss1 = -1 * activations.features[0, 0].mean()
    loss2 = mse(img_tensor, max_activation_input)
    loss = loss1 + loss2 * 100
    # print(activations.features[0, 0])

    print(f'loss:{loss.item():.4f}\tloss1:{loss1.item():.4f}\tloss2:{loss2.item():.4f}')

    loss.backward()
    # print(f'grad:{img_tensor.grad}')
    optimizer.step()
print(img_tensor)

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)
image = quantize(img_tensor, 1).cpu().detach().numpy()[0].transpose(1,2,0)
print(image)
plt.imshow(image)
time.sleep(10)
print(max_index,max_value)
fasfa,fasasf = testloader.dataset[max_index]
print(fasasf)
fasfa.shape
fasfa
print(fasfa)
plt.imshow(fasfa.cpu().detach().numpy().transpose(1,2,0))
time.sleep(10)