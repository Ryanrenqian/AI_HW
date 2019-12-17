import torch
import torch.nn as nn
import torch.optim as optim
from    torch.utils.data import DataLoader
from    torchvision import datasets
from    torchvision import transforms
from resnet import ResNet18
import os
from visdom import Visdom


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数设置
EPOCH = 135                              #遍历数据集次数
pre_epoch = 0                            #定义已经遍历数据集的次数
BATCH_SIZE = 128                         #批处理尺寸(batch_size)
LR = 0.01                                #学习率
modelpath="./model"                      #模型存放位置
isExists = os.path.exists(modelpath)
if not isExists:
    os.makedirs(modelpath)
# 准备数据集并预处理
cifar_train = datasets.CIFAR10('./data', True, transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
]), download=True)
cifar_train = DataLoader(cifar_train, batch_size=BATCH_SIZE, shuffle=True)

cifar_test = datasets.CIFAR10('./data', False, transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
]), download=True)
cifar_test = DataLoader(cifar_test, batch_size=BATCH_SIZE, shuffle=True)


# 模型定义-ResNet
net = ResNet18().to(device)

# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)



viz = Visdom()

viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
viz.line([0.], [0.], win='train_acc', opts=dict(title='train acc'))
viz.line([0.], [0.], win='test_acc', opts=dict(title='test acc'))
viz.line([0.], [0.], win='test_loss', opts=dict(title='test loss'))
if __name__ == "__main__":
    lr_list = []
    iterations = 0
    epoch=0
    while iterations<10000:

        net.train()
        epoch = epoch + 1

        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(cifar_train):
            iterations += 1

            # 准备数据
            length = len(cifar_train)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # forward + backward
            outputs = net(inputs)[0]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 每训练一次打印一次loss和准确率
            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            train_acc=100. * correct / total
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                  % (epoch, iterations, sum_loss / (i + 1), train_acc))
            viz.line([loss.item()], [iterations], win='train_loss', update='append')
            viz.line([train_acc], [iterations], win='train_acc', update='append')
            if iterations==10000:
                torch.save(net.state_dict(), '%s/net_%03d.pth' % (modelpath, iterations))
                break

        # 每训练完一个epoch测试一下准确率
        print("Waiting Test!")
        with torch.no_grad():
            test_correct = 0
            test_loss=0
            for data in cifar_test:
                net.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs,_ = net(images)
                test_loss += criterion(outputs, labels).item()
                pred = outputs.argmax(dim=1)
                test_correct += pred.eq(labels).float().sum().item()
            test_loss=test_loss/len(cifar_test.dataset)
            test_acc=test_correct / len(cifar_test.dataset)
            print(total)
            print(len(cifar_test.dataset))
            print('测试分类准确率为：%.3f%%' % (test_acc*100))
            print('测试分类loss为：%.03f' % (test_loss))
            viz.line([test_acc], [iterations], win='test_acc', update='append')
            viz.line([test_loss], [iterations], win='test_loss', update='append')
            print('Saving model......')
            torch.save(net.state_dict(), '%s/net_%03d.pth' % (modelpath, iterations))

