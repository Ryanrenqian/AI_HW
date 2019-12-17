import torch
import torchvision
import torchvision.transforms as transforms


def loadData(opt):
    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False,
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False,
                                           transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=2)

    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader


def load_testset(opt):
    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False,
                                           transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=1)
    return testloader
