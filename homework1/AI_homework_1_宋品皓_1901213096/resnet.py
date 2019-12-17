# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:08:08 2019

@author: Dell
"""

import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pylab

from tensorboardX import SummaryWriter


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg , (1 , 2 , 0)))
    pylab.show()


class residueBlock(nn.Module):
    def __init__(self,inplane,outplane,kernel_size=3,stride=1,padding=1):
        super(residueBlock, self).__init__()
        self.inplane = inplane
        self.outplane = outplane
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding 
        if inplane == outplane:
            self.dimension_inc = False
        else:
            self.dimension_inc = True
        self.conv1 = nn.Conv2d(self.inplane ,self.outplane ,kernel_size = self.kernel_size,stride=self.stride,padding=self.padding,bias=False)
        self.bn1 = nn.BatchNorm2d(self.outplane)
        
        self.conv2 = nn.Conv2d(self.outplane ,self.outplane ,kernel_size = self.kernel_size,stride=self.stride,padding=self.padding,bias=False)
        self.bn2 = nn.BatchNorm2d(self.outplane)
        self.projection_shortcut = nn.Conv2d(self.inplane ,self.outplane,1,stride=self.stride, bias=False)
        return
    def forward(self,x):
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        if self.dimension_inc == True:
            shortcut = self.projection_shortcut(shortcut)
            
        x = x + shortcut
        x = F.relu(x)
        return x
        
class Resnet(nn.Module):
    def __init__(self):
        super(Resnet,self).__init__()
        self.conv1 = nn.Conv2d(3 ,64 ,7,stride=2,padding=3)
        
        self.maxpool1 = nn.MaxPool2d(3,stride=2,padding=1)
        self.block1 = residueBlock(64,64)
        self.block2 = residueBlock(64,64)
        
        self.maxpool2 = nn.MaxPool2d(3,stride=2,padding=1)
        self.block3 = residueBlock(64,128)
        self.block4 = residueBlock(128,128)
        
        self.maxpool3 = nn.MaxPool2d(3,stride=2,padding=1)
        self.block5 = residueBlock(128,256)
        self.block6 = residueBlock(256,256)
        
        
        self.maxpool4 = nn.MaxPool2d(3,stride=2,padding=1)
        self.block7 = residueBlock(256,512)
        self.block8 = residueBlock(512,512)

        self.avepool1 = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 , 10)
        
        return
    
    def forward(self,x):
        x = self.conv1(x)

        x = self.maxpool1(x)
        x = self.block1(x)
        x = self.block2(x)
        
        x = self.maxpool2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = self.maxpool3(x)
        x = self.block5(x)
        x = self.block6(x)
        
        x = self.maxpool4(x)
        x = self.block7(x)
        x = self.block8(x)
        
        x = self.avepool1(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.softmax(x,dim=1)
        return x
    
def train(net,epochs,trainloader,testloader,classes,device_ids):
    writer = SummaryWriter(comment='resnet')

    # read model
    print('===> Try resume from checkpoint')
    if os.path.isdir('checkpoint'):
        try:
            checkpoint = torch.load('./checkpoint/resnet_final.t7')
            net.load_state_dict(checkpoint['state'])        
            start_epoch = checkpoint['epoch']
            print('===> Load last checkpoint data')
        except FileNotFoundError:
            start_epoch = 0
            print('Can\'t found resnet_final.t7')
    else:
        start_epoch = 0
        print('===> Start from scratch')
        
#    start_epoch = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters() , lr = 0.005,momentum=0.9)
#    optimizer = nn.DataParallel(optimizer, device_ids=device_ids)
#    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.3)
    
    #train the Network
    for epoch in range(epochs):
        running_loss = 0.0
        epoch_loss = 0.0
        if epoch + start_epoch > 10000:
            break
        correct = 0
        total = 0
        for i , data in enumerate(trainloader , 0):
            inputs , labels = data
            inputs , labels = Variable(inputs.cuda()) , Variable(labels.cuda())
#            inputs = normalize1d(inputs,0.5,0.5)
            optimizer.zero_grad()
            #forward + backward + optimizer
            outputs = net(inputs)
            _ , predicted = torch.max(outputs.data , 1)
            correct += (predicted == labels.cuda()).sum()
            total += labels.size(0)
            loss = criterion(outputs , labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.data
            epoch_loss = running_loss /(i+1)
            print('[%d %d] loss: %.3f' % (epoch+start_epoch+ 1 , i+1 ,running_loss /(i+1)))
        train_acc = float(correct) / total
#        print(train_acc)

    
        # validation
        correct = 0
        total = 0
        for data in testloader:
            images , labels = data
            outputs = net(Variable(images.cuda()))
            _ , predicted = torch.max(outputs.data , 1)
            correct += (predicted == labels.cuda()).sum()
            total += labels.size(0)
        val_acc = float(correct) / total
#        print(val_acc)
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
        
        if epoch % 100 == 99:
            class_correct = torch.ones(10).cuda()
            class_total = torch.ones(10).cuda()
            for data in testloader:
                images , labels = data
                outputs = net(Variable(images.cuda()))
                _ , predicted = torch.max(outputs.data , 1)
                c = (predicted == labels.cuda()).squeeze()
                #print(predicted.data[0])
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i]
                    class_total[label] += 1
            
            for i in range(10):
#                writer.add_scalar('Accuracy of '+classes[i],100 * class_correct[i] / class_total[i],epoch+start_epoch)
                print('Accuracy of %5s : %2d %%' % (classes[i] , 100 * class_correct[i] / class_total[i]))
            
        
        # save training loss and accuracy
        writer.add_scalar('cross_entropy',epoch_loss,epoch+start_epoch)
        writer.add_scalar('train_accuracy',train_acc,epoch+start_epoch)
        writer.add_scalar('validation_accuracy',val_acc,epoch+start_epoch)
        
        print('epoch[%d] loss: %.3f' % (epoch+start_epoch+ 1 , running_loss))
        if epoch % 10 == 9:
            print('===> Saving models...')
            state = {
                'state': net.state_dict(),
                'epoch': epoch+start_epoch                           }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/resnet_final.t7')
        
#    writer.add_graph(net,(inputs,))
    writer.close()
    print('Finished Training')
    
    



   


if __name__=='__main__':
    transform = transforms.Compose([transforms.RandomResizedCrop(224),transforms.ToTensor() ,transforms.Normalize((0.5 , 0.5 , 0.5) , (0.5 , 0.5 , 0.5))])  
    trainset = torchvision.datasets.CIFAR10(root = './data' , train = True , download = True , transform = transform)
    trainloader = torch.utils.data.DataLoader(trainset , batch_size = 512 , shuffle = True , num_workers =2)    
    testset = torchvision.datasets.CIFAR10(root = './data' , train = False , download = True , transform = transform)
    testloader = torch.utils.data.DataLoader(testset , batch_size = 128 , shuffle = True , num_workers = 2)
    classes = ('plane' , 'car' , 'bird' , 'cat' , 'deer' , 'dog' , 'frog' , 'horse' , 'ship' , 'truck')
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"
    
    resnet = Resnet()
    device_ids=[0,1,2]
    resnet = resnet.cuda(device_ids[0])
    net=torch.nn.DataParallel(resnet,device_ids=device_ids)
    train(net,200,trainloader,testloader,classes,device_ids)
    
    
    

    
    
