import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transf
from time import time
from tensorboardX import SummaryWriter
import res18
import view_lib
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import os
import utils

BATCH_SIZE = 64
EPOCH = 300
lr = 0.01

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter()

net = res18.ResNet18()

Trans = transf.Compose([
    transf.Resize(224, interpolation=4),
    transf.RandomHorizontalFlip(),
    transf.ToTensor(),
    transf.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
Test = transf.Compose([
    transf.Resize(224, interpolation=4),
    transf.ToTensor(),
    transf.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
if __name__ == '__main__':
    train_data = torchvision.datasets.CIFAR10(root='../data', train=True, transform=Trans, download=False)
    test_data = torchvision.datasets.CIFAR10(root='../data', train=False, transform=Test, download=False)
    train_batch = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_batch = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)  # 权重衰减
    model = net.to(device)
    best_acc = 60
    utils.prepare_dir('./train_data', empty=False)
    with open("./train_data/acc.txt", "w") as f:
        with open("./train_data/log.txt", "w")as f2:
            for epoch in range(1, EPOCH):
                if epoch > 60:
                    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.5,
                                                weight_decay=5e-4)  # 权重衰减
                starttime = time()
                print('\nEpoch: %d start:' % epoch)
                net.train()
                sum_loss = 0.0
                correct = 0.0
                num = 0.0
                for step, data in enumerate(train_batch, 0):
                    # 准备数据
                    length = len(train_batch)
                    features, labels = data
                    # print(features.size())
                    features, labels = features.to(device), labels.to(device)
                    optimizer.zero_grad()
                    output = net(features)

                    loss = criterion(output, labels)
                    loss.backward()
                    optimizer.step()
                    # sum_loss += loss.item()
                    train_iter = step + 1 + (epoch - 1) * length

                    # num += labels.size(0)   #训练集准确率
                    # _, preout = torch.max(output.data, 1)  # 返回分类结果
                    # correct += preout.eq(labels.data).sum()
                    # acc = 100. * correct / num
                    # writer.add_scalar('scaler/acc', acc, train_iter)
                    # writer.add_scalar('scaler/loss', loss.item(), train_iter)
                    # print('[epoch:%d, step:%d] Loss: %.03f | Acc: %.3f%% '
                    #       % (epoch, train_iter, sum_loss / (step + 1), 100. * correct / num))
                    # f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                    #          % (epoch,train_iter, sum_loss / (step + 1), 100. * correct / num))
                    # f2.write('\n')
                    # f2.flush()

                # test for epoch
                print("Test!")
                with torch.no_grad():
                    correct = 0
                    num = 0
                    for data in test_batch:
                        net.eval()
                        features, labels = data
                        features, labels = features.to(device), labels.to(device)
                        output = net(features)
                        _, preout = torch.max(output.data, 1)
                        num += labels.size(0)
                        correct += (preout == labels).sum()
                    acc = 100. * correct / num
                    writer.add_scalar('scaler/acc', acc, train_iter)
                    writer.add_scalar('scaler/loss', loss.item(), train_iter)
                    print('test_acc：%.3f%%' % acc)
                    print('time:%.3f' % (time() - starttime))
                    #                print('Saving model......')
                    #                torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))
                    torch.save(net.state_dict(), './net/resnet18.pth')  # 保存模型
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch, acc))
                    f.write('\n')
                    f.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if acc > best_acc:
                        f3 = open("./train_data/best_acc.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch, acc))
                        f3.close()
                        best_acc = acc
            print("Training Finished, TotalEPOCH=%d" % EPOCH)
    writer.close()

