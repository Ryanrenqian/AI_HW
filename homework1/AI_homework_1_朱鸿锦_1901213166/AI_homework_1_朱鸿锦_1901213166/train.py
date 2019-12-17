# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 00:29:39 2019

@author: acrobat
"""
import os
import sys
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from torchvision import utils
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
#from dataset import *
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from model import Resnet,BasicBlock

#关于梯度更新 https://blog.csdn.net/byron123456sfsfsfa/article/details/90609758
def train(epoch):
    
    net.train()
    #net 哪里有train这个方法的
    for batch_index, (images, lables) in enumerate(cifar10_training_loader):
        images = Variable(images)
        lables = Variable(lables)
        
        lables = lables.cuda() #意思就是把变量放在cuda上面
        images = images.cuda()
        
        optimizer.zero_grad()#把梯度置0，方便后续计算梯度
        outputs = net(images)
        loss = loss_function(outputs, lables)
        loss.backward() #这一步是求导
        optimizer.step() #这一步是更新参数
        
        n_iter = (epoch -1)*len(cifar10_training_loader) +batch_index +1 #总共的迭代次数
        
        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR:{:0.6f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.b + len(images),
                total_samples=len(cifar10_training_loader.dataset)))
        
        writer.add_scalar('Train/loss',loss.item(),n_iter)



def eval_training(epoch):
    net.eval()
    #这个的意思是实现net.eval这个方法么，不是 好像是调用测试模型，这样数据输入网络时候可能bn之类的和训练不同
    
    test_loss = 0.0
    correct = 0.0
    
    for(images,lables) in cifar10_test_loader:
        images = Variable(images)
        lables = Variable(lables)
        
        images = images.cuda()
        lables = lables.cuda()
    
        outputs = net(images)
        loss = loss_function(outputs,lables)
        test_loss += loss.item() #我们在提取 loss 的纯数值的时候，常常会用到 loss.item()，其返回值是一个 Python 数值 (python number)
        _, preds = outputs.max(1)#最大那个是预测结果，看预测结果准不准， 但是这个_,pred 是个什么意思
        correct += preds.eq(lables).sum()
        
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
            test_loss / len(cifar10_test_loader.dataset),
            correct.float() / len(cifar10_test_loader.dataset)
            ))
    print()
    
    #add information to tensorboard
    #先输入进去标题‘test/acc’ 然后是y值，然后是x坐标
    writer.add_scalar('Test/Average loss', test_loss / len(cifar10_test_loader.dataset), epoch)
    writer.add_scalar('Test/Accuracy', correct.float() / len(cifar10_test_loader.dataset), epoch)
    
    return correct.float() / len(cifar10_test_loader.dataset)    
        
def get_training_dataloader(batch_size=16, shuffle=True):

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])
    cifar10_training = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    # 有关dataloader的介绍 https://www.cnblogs.com/ranjiewen/p/10128046.html
    cifar10_training_loader = DataLoader(
        cifar10_training, shuffle=shuffle, batch_size=batch_size)

    return cifar10_training_loader
   

def get_test_dataloader(batch_size=16,shuffle=True):


    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    cifar10_test = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    cifar10_test_loader = DataLoader(
        cifar10_test, shuffle=shuffle,  batch_size=batch_size)

    return cifar10_test_loader


def visTensor(tensor,epoch,name, ch=0, allkernels=False, nrow=8, padding=1): 
    n,c,w,h = tensor.shape #获取tensor的各个维度信息，64 3 7 7

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1) #如果这个滤镜不是3维的，那么压缩到一维

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)), cmap = plt.get_cmap('gray_r'))
    plt.savefig(os.path.join('./visTensor3','epoch{}-{}.jpg'.format(epoch,name)))
    #plt.imsave(os.path.join('./visTensor','epoch{}-{}.jpg'.format(epoch,name)),grid.numpy().transpose((1, 2, 0))) #重置了figure的分辨率，所以显示出来大，如果直接保存，分辨率太低    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-batchsize',type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-a',type = bool, default=False, help='test the filter, reconstructed')
    parser.add_argument('-b',type = bool, default=False, help='test the acc')
    parser.add_argument('-c',type = bool, default=True, help='train')
    args = parser.parse_args()
    
    net = Resnet(BasicBlock)
    net = net.cuda() #参数和模型应该都放在cuda上
    
    cifar10_training_loader = get_training_dataloader(
            batch_size=args.batchsize,
            shuffle=args.s)
    
    cifar10_test_loader = get_test_dataloader(
            batch_size=args.batchsize,
            shuffle=args.s)
    
    cifar10_image_loader = get_test_dataloader(
            batch_size =1,
            shuffle=args.s)
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    #TIME_NOW = datetime.now().isoformat() #这个文件命名有点问题，Windows下文件夹名称不允许有：
    TIME_NOW = '20191025'
    checkpoint_path = os.path.join('checkpoint',TIME_NOW)
    
    #为checkpoint创建存储空间
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path,'{epoch}-{type}.pth')
    #这里的epoch和type是哪里的 而且那个。pyh是文件类型么
    
    #tensorboard的尝试 我要用的不是graph 这个还需要改 https://www.aiuai.cn/aifarm646.html
    #if not os.path.exists('tensorboard_runs'):
    #    os.mkdir('tensorboard_runs')
    writer = SummaryWriter()
#    input_tensor = torch.Tensor(12,3,32,32).cuda()#这是啥！！！大概是输入网络的参数？
#    writer.add_graph(net, Variable(input_tensor, requires_gard=True))
    
    #训练完成后重构效果
    a = args.a
    if a:
        net.load_state_dict(torch.load(r'C:\Users\acrobat\.spyder-py3\checkpoint\20191021\123-best.pth'),True)
                    #将kernels可视化
        epoch = 123
        kernels = net.conv1[0].weight.cpu().data.clone()
        visTensor(kernels,epoch,'conv1-0filter', ch=0, allkernels=False)
            
        for n_iter, (image, label) in enumerate(cifar10_image_loader):
            if n_iter > 0:
                break
                #将测试图片可视化
            visTensor(image,epoch,'testimage', ch=0, allkernels=False)
                #重构conv164
            for i in range(0,64):
                    #一层一层得到重建图conv1
                kernels = net.get_reftconv1(image,i).data.clone()
                visTensor(kernels,i,'conv1-reconstructed', ch=0, allkernels=False)
                #重构conv5512
            for i in range(0,512):
                    #一层一层得到重建图conv1
                kernels = net.get_reftconv5(image,i).data.clone()
                visTensor(kernels,i,'conv5-reconstructed', ch=0, allkernels=False)
                
                #将图片输入conv1里，得到64个feature maps
            kernels = net.get_featureconv1(image).cpu().data.clone()
            visTensor(kernels,epoch,'conv1-0feature', ch=0, allkernels=True)

            plt.axis('off')
            plt.ioff()
            plt.show()
    #尝试从checkpoint读取数据并继续训练
    b = args.b
    if b:
        net.load_state_dict(torch.load(r'C:\Users\acrobat\.spyder-py3\checkpoint\20191021\123-best.pth'),True)
        #加r的意思是只读路径，没有什么转义字符；net把最好的网络读进来之后，可以集训训练或者测试结果
        net.eval()#开启评测模式

        correct_1 = 0.0
        correct_5 = 0.0
        total = 0

        for n_iter, (image, label) in enumerate(cifar10_test_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar10_test_loader)))
            image = Variable(image).cuda()
            label = Variable(label).cuda()
            output = net(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

        #compute top 5
            correct_5 += correct[:, :5].sum()

        #compute top1 
            correct_1 += correct[:, :1].sum()


        print()
        print("Top 1 err: ", 1 - correct_1 / len(cifar10_test_loader.dataset))
        print("Top 5 err: ", 1 - correct_5 / len(cifar10_test_loader.dataset))
        print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
    c = args.c
    if c:
        best_acc = 0.0 #到了一定次数之后记录最好结果而不是最后结果
        for epoch in range(1,200):
            train(epoch)
            acc = eval_training(epoch)
        
            if epoch > 60 and best_acc < acc:
                torch.save(net.state_dict(), checkpoint_path.format(epoch = epoch, type = 'best'))
                best_acc = acc
                continue
        
            if not epoch % 20:
                torch.save(net.state_dict(),checkpoint_path.format(epoch=epoch, type='regular'))
#            #将kernels可视化
#            kernels = net.conv1[0].weight.cpu().data.clone()
#            visTensor(kernels,epoch,'conv1-0filter', ch=0, allkernels=False)
#            
#            for n_iter, (image, label) in enumerate(cifar10_image_loader):
#                if n_iter > 1:
#                    break
#                #将测试图片可视化
#                visTensor(image,epoch,'testimage', ch=0, allkernels=False)
#                #重构conv164
#                for i in range(0,64):
#                    #一层一层得到重建图conv1
#                    kernels = net.get_reftconv1(image,i).data.clone()
#                    visTensor(kernels,i,'conv1-reconstructed', ch=0, allkernels=False)
#                #重构conv5512
#                for i in range(0,512):
#                    #一层一层得到重建图conv1
#                    kernels = net.get_reftconv5(image,i).data.clone()
#                    visTensor(kernels,i,'conv5-reconstructed', ch=0, allkernels=False)
#                
#                #将图片输入conv1里，得到64个feature maps
#                kernels = net.get_featureconv1(image).cpu().data.clone()
#                visTensor(kernels,epoch,'conv1-0feature', ch=0, allkernels=True)
#
#            plt.axis('off')
#            plt.ioff()
#            plt.show()
    
    writer.close()
            
    
        
