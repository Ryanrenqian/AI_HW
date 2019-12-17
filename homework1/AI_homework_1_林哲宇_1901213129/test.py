import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import Myvisdom
import MyResnet

# Parameters

Epoch = 25
Batch_size = 128
LR = 0.01 

# Visdom

vidsomline = Myvisdom.VisdomLine(env_name='Resnet_VisdomLine')

# Datasets CIFAR10

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform = transforms.Compose([transforms.Resize((224,224)), \
    transforms.ToTensor(), \
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=Batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Load Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = MyResnet.ResNet(MyResnet.ResBlock).to(device)

# Loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = LR, momentum = 0.9, weight_decay=5e-4)
best_val = 0.0

# Train and Test

for epoch in range(Epoch):
    net.train()
    print("Training")
    running_loss = 0.0
    accuracy = 0.0
    counter = 0.0
    for i, data in enumerate(trainloader, 0):
        length = len(trainloader)
        # print(length)
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs, conv1_output, conv5_output = net(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        counter += labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        accuracy += (predicted == labels).sum().item()
        print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), running_loss / (i + 1), 100. * accuracy / counter))
    vidsomline.plot('loss', 'train', 'Train Loss', epoch, running_loss / len(trainloader))

    net.eval()
    print("Validation")
    with torch.no_grad():
        correct = 0
        total = 0
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs, conv1_output, conv5_output = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100. * correct / total
        print("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, accuracy))
        vidsomline.plot('acc', 'val', 'Validation Accuracy', epoch, accuracy)

        if accuracy > best_val:
            f = open("best_val.txt", "w")
            f.write("EPOCH=%d,best_val= %.3f%%" % (epoch + 1, accuracy))
            f.close()
            best_val = accuracy

print('Finished Training')

PATH = './cifar_net_224.pth'
torch.save(net.state_dict(), PATH)
