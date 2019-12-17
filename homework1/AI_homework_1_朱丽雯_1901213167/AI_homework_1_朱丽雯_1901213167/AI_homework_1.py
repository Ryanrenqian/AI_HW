'''
zhuliwen: liwenzhu@pku.edu.cn
October 24，2019
ref: https://github.com/weiaicunzai/pytorch-cifar100
'''


import os
import argparse
from datetime import datetime
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torchsummary import summary
from torch.optim.lr_scheduler import _LRScheduler

class Settings:
    def __init__(self):
        self.CIFAR10_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        self.CIFAR10_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

        self.CIFAR10_TEST_MEAN = (0.5088964127604166, 0.48739301317401956, 0.44194221124387256)
        self.CIFAR10_TEST_STD = (0.2682515741720801, 0.2573637364478126, 0.2770957707973042)

        # directory to save weights file
        self.CHECKPOINT_PATH = 'save_weights'

        # total training epoches
        self.EPOCH = 26
        self.MILESTONES = [5, 10, 15, 20, 25]

        # initial learning rate
        # INIT_LR = 0.1

        # time of we run the script
        self.TIME_NOW = datetime.now().isoformat()

        # tensorboard log dir
        self.LOG_DIR = 'tensorboard'

        # save weights file per SAVE_EPOCH epoch
        self.SAVE_EPOCH = 1

settings = Settings()

def get_training_dataloader(mean, std, batch_size=128, num_workers=2, shuffle=True):

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        # transforms.Resize(size=(224, 224)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar10_training = CIFAR10Train(path, transform=transform_train)
    cifar10_training = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    cifar10_training_loader = DataLoader(
        cifar10_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar10_training_loader

def get_test_dataloader(mean, std, batch_size=128, num_workers=2, shuffle=True):

    transform_test = transforms.Compose([
        # transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar10_test = CIFAR10Test(path, transform=transform_test)
    cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    cifar10_test_loader = DataLoader(
        cifar10_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar10_test_loader

class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=10):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv2_x = self._make_layer(block, 64, num_block[0], 2)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

def train(epoch):

    net.train()

    for batch_index, (images, labels) in enumerate(cifar10_training_loader):
        start_batch_time = time.time()

        correct = 0.0

        if epoch <= args.warm:
            warmup_scheduler.step()

        images = Variable(images)
        labels = Variable(labels)

        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()




        n_iter = (epoch - 1) * len(cifar10_training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        end_batch_time = time.time()
        time_spent = end_batch_time - start_batch_time

        print('Training Epoch: {epoch} [{trained_iter}/{total_iter}]\tAccuracy: {:0.4f}\tLoss: {:0.4f}\tLR: {:0.6f}\tTime: {:0.4f}'.format(
            correct.float() / len(labels),
            loss.item(),
            optimizer.param_groups[0]['lr'],
            time_spent,
            epoch=epoch,
            trained_iter=n_iter,
            total_iter= settings.EPOCH * len(cifar10_training_loader)
        ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)
        writer.add_scalar('Train/Accuracy', correct.float() / len(labels), n_iter)

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

def eval_training(epoch):
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in cifar10_test_loader:
        images = Variable(images)
        labels = Variable(labels)

        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(cifar10_test_loader.dataset),
        correct.float() / len(cifar10_test_loader.dataset)
    ))
    print()

    #add informations to tensorboard
    writer.add_scalar('Test/Average loss', test_loss / len(cifar10_test_loader.dataset), epoch)
    writer.add_scalar('Test/Accuracy', correct.float() / len(cifar10_test_loader.dataset), epoch)

    return correct.float() / len(cifar10_test_loader.dataset)


if __name__ == '__main__':
    global_begin_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='resnet18', help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    args = parser.parse_args()



    net = ResNet(BasicBlock, [2, 2, 2, 2])
    net = net.cuda()
    summary(net, input_size=(3, 32, 32))
    print(net)

    # data preprocessing:
    cifar10_training_loader = get_training_dataloader(
        settings.CIFAR10_TRAIN_MEAN,
        settings.CIFAR10_TRAIN_STD,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )

    cifar10_test_loader = get_test_dataloader(
        settings.CIFAR10_TEST_MEAN,
        settings.CIFAR10_TEST_STD,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES,
                                                     gamma=0.2)  # learning rate decay
    iter_per_epoch = len(cifar10_training_loader)


    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    # use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(
        settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(12, 3, 32, 32).cuda()
    writer.add_graph(net, Variable(input_tensor, requires_grad=True))

    # create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    for epoch in range(1, settings.EPOCH+1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        train(epoch)
        acc = eval_training(epoch)

        # start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))

    writer.close()
    global_end_time = time.time()
    print('End, the total training time spent: ', global_end_time - global_begin_time)

