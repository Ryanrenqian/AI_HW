import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
writer = SummaryWriter('./log')

import argparse
import os

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(2019)


# transform = transforms.Compose(
#     [transforms.RandomCrop(32, padding=4),  # padding 0，and convert the image to 32*32
#      transforms.RandomHorizontalFlip(),     # flip half of the image
#      transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # mean and variance of RGB normalize

def transform(x):
    x = x.resize((224, 224), 2) # resize the image from 32*32 to 224*224
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5 # Normalize
    x = x.transpose((2, 0, 1)) # reshape, put the channel to 1-d;  input = {channel, size, size}
    x = torch.from_numpy(x)
    return x
trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),  # inplace = True
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        if stride == 1 and inchannel == outchannel:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
    def forward(self, x):
        return F.relu(self.left(x) + self.shortcut(x))


class ResNet18(nn.Module):
    def __init__(self, ResidualBlock, num_class=10):
        super(ResNet18, self).__init__()
        # First Conv Layer: kenel=7, channel=64, stride=2, and feature map size is halved
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 64, stride=1)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128, stride=1)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256, stride=1)
        )
        self.layer4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512, stride=1)
        )
        self.fc = nn.Linear(512, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = F.max_pool2d(out, 3, stride=2, padding=1)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 7)  ####
        out = out.view(out.size(0), -1)  # 1,4608
        out = self.fc(out)
        return out



# 超参数设置
EPOCH = 60   #遍历数据集次数
pre_epoch = 0
BATCH_SIZE = 64      #批处理尺寸(batch_size)
LR = 0.01        #学习率

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)    # lr = 0.0001

net = ResNet18(ResidualBlock).to(device)

criterion = nn.CrossEntropyLoss()  # use cross entropy as loss function
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4) # optimizer = mini-batch momentum-SGD，use L2 Norm()
# 并采用L2正则化（权重衰减）



def visual_filter():
    path = '/home/hzq/tmp/waibao/models/net_003.pth'
    checkpoint = torch.load(path)
    model = ResNet18(ResidualBlock)
    model.load_state_dict(checkpoint)
    model.eval()
    images, labels = iter(trainloader).next()
    images, labels = images.to(device), labels.to(device)
    # img = np.uint8(np.random.uniform(150, 180, (sz, sz, 3)))/255


# 训练
if __name__ == "__main__":

    # filter visualization
    # visual_filter()
    # assert 0==1

    best_acc = 80  #2 init best test accuracy
    print("Start Training Resnet18")
    with open("acc.txt", "w") as f:
        with open("log.txt", "w")as f2:
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(trainloader, 0):
                    # 准备数据
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # forward + backward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()


                    # 每训练1个batch打印一次loss和准确率(即一个iteration)
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    if i % 200 == 0:
                        print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                              % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()

                    # tensorboardX
                    niter = epoch * length + i
                    writer.add_scalars('/home/hzq/tmp/waibao/' + 'Train_loss',
                                       {'train_loss': loss.data.item()}, niter)
                    writer.add_scalars('/home/hzq/tmp/waibao/' + 'Train_loss_2',
                                       {'train_loss_2': sum_loss / (i + 1)}, niter)
                    writer.add_scalars('/home/hzq/tmp/waibao/' + 'Train_Acc',
                                       {'train_acc': 100. * predicted.eq(labels.data).cpu().sum() / labels.size(0)}, niter)


                # acc in test set per epoch
                print("Waiting Test!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in testloader:
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    print('测试分类准确率为：%.3f%%' % (100 * correct / total))
                    acc = 100. * correct / total
                    # 将每次测试结果实时写入acc.txt文件中
                    print('Saving model......')
                    torch.save(net.state_dict(), '%s/net_%03d.pth' % ('/home/hzq/tmp/waibao/models/', epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if acc > best_acc:
                        f3 = open("best_acc.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
            print("Training Finished, TotalEPOCH=%d" % EPOCH)


