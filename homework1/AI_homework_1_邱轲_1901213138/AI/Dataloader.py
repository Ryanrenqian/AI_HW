import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms


Cifar10_train_mean = (0.4914, 0.4822, 0.4465)
Cifar10_train_std = (0.2023, 0.1994, 0.2010)

# Not Resize to 224
def Load_Cifar10(batch_size):

    # Data augmentation  Resize to 224  use: transforms.Resize(224)
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
                                         transforms.ToTensor(),
                                         transforms.Normalize(Cifar10_train_mean, Cifar10_train_std), ])
    train_dataset = torchvision.datasets.CIFAR10(root='./data/', train=True, download=False, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    transform_test = transforms.Compose([ transforms.ToTensor(),
                                         transforms.Normalize(Cifar10_train_mean,
                                         Cifar10_train_std), ])
    test_dataset = torchvision.datasets.CIFAR10(root='./data/', train=False, download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    print("End of Load Cifar10!")

    return train_loader, test_loader

