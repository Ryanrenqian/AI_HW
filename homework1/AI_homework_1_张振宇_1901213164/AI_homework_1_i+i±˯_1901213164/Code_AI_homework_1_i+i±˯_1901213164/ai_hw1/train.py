def train():
    import torch
    import torchvision
    import torchvision.transforms as transforms
    import zzytools
    torch.backends.cudnn.benchmark = True
    #变换技巧来自于https://blog.csdn.net/weixin_40123108/article/details/85246784
    transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transform_test = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


    trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=24)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=24)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    from ZZYResNet18 import ZZYResNet18
    resnet18 = ZZYResNet18(n_classes=10)

    import torch.optim as optim
    import torch.nn as nn
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = resnet18
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4, amsgrad=False)

    from torch.utils.tensorboard import SummaryWriter
    import numpy as np

    writer = SummaryWriter(flush_secs=20)
    nth_minibatch = 0
    running_loss = 0.0
    print(net)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    epoch_timer = zzytools.TimerFactory.produce('epoch_time')
    train_timer = zzytools.TimerFactory.produce('training_time')
    test_timer = zzytools.TimerFactory.produce('testing_time')
    for epoch in range(100):  # loop over the dataset multiple times
        epoch_timer.start()
        train_timer.start()
        net.train()
        print('lr:{}'.format(optimizer.param_groups[0]['lr']))

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            with torch.no_grad():
                _, batch_predicted = torch.max(outputs.detach(), 1)
                batch_size = labels.size(0)
                batch_correct = (batch_predicted == labels).sum().item()
                writer.add_scalar('Accuracy/train', batch_correct / batch_size, nth_minibatch)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            writer.add_scalar('Loss/train', loss.item(), nth_minibatch)

            if nth_minibatch % 100 == 99:  # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, nth_minibatch + 1, running_loss / 100))
                running_loss = 0.0
            nth_minibatch += 1
        train_timer.end()
        test_timer.start()
        correct = 0
        total = 0
        with torch.no_grad():
            net.eval()
            test_total_loss = 0
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                test_total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()


        print('Accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct / total))
        writer.add_scalar('Accuracy/test', 100 * correct / total, epoch)
        writer.add_scalar('Loss/test', test_total_loss, epoch)
        scheduler.step()
        PATH = './cifar_net.pth'
        torch.save(net.state_dict(), PATH)
        print('Model Saved')
        test_timer.end()
        epoch_timer.end()
        print(f'Epoch: {epoch_timer.get_accumulated_time()}s\tTraining: {train_timer.get_accumulated_time()}s\tTesting: {test_timer.get_accumulated_time()}s')





if __name__ == '__main__':
    train()
