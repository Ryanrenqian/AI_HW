
# set imports
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from resnet import *
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from utils import *
import os

def train():
    # data loading
    transform_train = transforms.Compose([
        # transforms.Resize(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # transforms.ColorJitter(),
        # transforms.Normalize((0.4914, 0.5, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        # transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='../data',
        train=True,
        download=False,
        transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=False,
        transform=transform_test
    )
    # DataLoaders
    batch_size = 32
    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=2)
    testloader = DataLoader(testset,batch_size=batch_size,
                            shuffle=False,num_workers=2)
    # constant for classes
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')
    # device choice
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # loading model
    expid = 'cifar10_resnet18_experiment_1'
    if not os.path.exists('runs/%s'%expid):
        os.makedirs('runs/%s'%expid,True)
    net = ResNet().to(device)
    writer = SummaryWriter(log_dir='runs/%s'%expid)
    # definition loss and optim
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.99))
    EPOCH = 60
    print("---------------Start Training, Resnet-18-------------")

    # start training
    for epoch in range(EPOCH):
        running_loss = 0.0
        correct = 0.0
        total = 0.0
        if epoch == 10:
            optimizer = optim.SGD(net.parameters(),lr=0.0001,momentum=0.9)
        for i, data in enumerate(trainloader, 0):
            length = len(trainloader)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            if i % 1000 == 999:
                print('[epoch:%d - iter:%d] Loss: %.03f | Acc: %.3f%% '
                      % (epoch + 1, (i + 1 + epoch * length), running_loss / (i + 1), 100. * correct / total))
                writer.add_scalar('train_loss',
                                  running_loss / 1000,
                                  epoch * length + i+1)
                writer.add_scalar('train_accuracy',
                                  100. * correct / total,
                                  epoch * length + i+1)
                running_loss = 0.0
        # 测试数据集
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                net.eval()
                inputs,labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total +=labels.size(0)
                correct +=(predicted == labels).cpu().sum()
            print('Classification accuracy in Evaluation ：%.3f%%' % (100. * correct / total))
            acc = 100. * correct / total
            writer.add_scalar('test accuracy',100. * correct / total,epoch * len(trainloader) + i)
        if epoch % 8 ==0:
            torch.save(net.state_dict(), 'runs/%s/%d_epoch_para.pkl'%(expid,epoch))
    print('Finished training')


if __name__ == '__main__':
    train()


