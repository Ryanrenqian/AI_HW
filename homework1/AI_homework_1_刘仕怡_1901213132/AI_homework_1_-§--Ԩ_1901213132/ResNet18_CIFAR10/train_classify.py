# coding with UTF-8
# ******************************************
# *****CIFAR-10 with ResNet8 in Pytorch*****
# *****train_classify.py               *****
# *****Author：Shiyi Liu               *****
# *****Time：  Oct 22nd, 2019          *****
# ******************************************
import torch
import torchvision
import numpy as np
import os
import time
import conv_visual
from network import ResNet18
import argparse
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
import warnings
warnings.filterwarnings("ignore")


TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
MILESTONES = [75, 150]
CHECKSTONES = [100, 500, 1000, 5000, 10000]
ITERS = 10000
SAVE_PATH = 'model/'
LOG_PATH = 'log/'
PLOT_PATH = 'plots/'


def load_mestd():
    return TRAIN_MEAN, TRAIN_STD


def load_data(mean, std, train_batchsize, test_batchsize, num_workers=2, shuffle=True):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    trainloader = DataLoader(dataset=train_set, shuffle=shuffle, num_workers=num_workers,
                              batch_size=train_batchsize)
    testloader = DataLoader(dataset=test_set, shuffle=shuffle, num_workers=num_workers,
                             batch_size=test_batchsize)
    return trainloader, testloader


def get_device():
    if torch.cuda.is_available():
        cudnn.benchmark = True
        dv = torch.device('cuda')
        print("cuda is on")
    else:
        dv = torch.device('cpu')

    return dv


def load_model():
    md = ResNet18().to(device)
    return md


def save(modeltype, model_out_path):
    model_out_path = os.path.join(model_out_path, '{modeltype}.pth')
    torch.save(model, model_out_path.format(modeltype=modeltype))
    print("Checkpoint saved to {}".format(model_out_path))


class warmup(_LRScheduler):
    def __init__(self, optimizer_warm, total_iters, last_epoch = -1):
        self.total_iters = total_iters
        super().__init__(optimizer_warm, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def test_training():
    model.eval()
    batch_loss = 0.0
    acc = 0.0
    total = 0.0

    print(len(test_loader))
    with torch.no_grad():
        for batch_num, (img, label) in enumerate(test_loader):
            img, label = img.to(device), label.to(device)

            output = model(img)
            loss = loss_function(output, label)
            batch_loss += loss.item()
            outlabel = torch.max(output, 1)
            total += label.size(0)
            acc += np.sum(outlabel[1].cpu().numpy() == label.cpu().numpy())

        print(batch_num)
        print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}% ({acc}/{total})'.format(
            batch_loss / len(test_loader),
            100.0 * acc / total,
            acc=int(acc),
            total=int(total)
        ))

    writer.add_scalar('Test/Average loss', batch_loss / len(test_loader), epoch)
    writer.add_scalar('Test/Accuracy', acc / len(test_loader.dataset), epoch)
    return batch_loss / len(test_loader), acc/total


def train():
    model.train()
    batch_loss = 0
    acc = 0
    total = 0.0

    for batch_num, (img, label) in enumerate(train_loader):
        if epoch <= args.warm:
            warm_sch.step()

        img, label = img.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(img)
        loss = loss_function(output, label)
        loss.backward()
        optimizer.step()
        batch_loss += loss.item()
        total += label.size(0)
        outlabel = torch.max(output, 1)
        total_iter = (epoch - 1) * len(train_loader) + batch_num +1
        acc += np.sum(outlabel[1].cpu().numpy() == label.cpu().numpy())

        writer.add_scalar('Train/loss', loss.item(), total_iter)
        writer.add_scalar('Train/Accuracy', 100. * acc / total, total_iter)
        

        last_layer = list(model.children())[-1]
    

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tIters: [{current_iter}/{total_iters}]\t\t'
              'Global Loss: {:0.4f}\tBatch Loss: {:0.4f}\tLR: {:0.6f}\tAcc: {:.4f}% ({acc}/{total})'.format(
            batch_loss/float(batch_num+1),
            loss.item(),
            optimizer.param_groups[0]['lr'],
            100. * acc / total,
            acc=acc,
            current_iter=total_iter,
            total_iters=ITERS,
            total=int(total),
            epoch=epoch,
            trained_samples=batch_num * args.trainBatchSize + len(img),
            total_samples=len(train_loader.dataset)
        ))

    return batch_loss / len(train_loader), acc/total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="cifar-10 with ResNet18 in PyTorch")
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epoch', default=20, type=int, help='number of epochs tp train for')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('--trainBatchSize', default=100, type=int, help='training batch size')
    parser.add_argument('--testBatchSize', default=100, type=int, help='testing batch size')
    parser.add_argument('--mean', default=TRAIN_MEAN, type=float, help='training batch size')
    parser.add_argument('--std', default=TRAIN_STD, type=float, help='training batch size')
    parser.add_argument('--pth', default=SAVE_PATH, help='training batch size')
    parser.add_argument('--log', default=LOG_PATH, help='training batch size')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    args = parser.parse_args()
    device = get_device()
    model = ResNet18().to(device)
    train_loader, test_loader = load_data(mean=args.mean, std=args.std,
                                          train_batchsize=args.trainBatchSize, test_batchsize=args.testBatchSize)


    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15, 18], gamma=0.5)
    iter_per_epoch = len(train_loader)
    warm_sch = warmup(optimizer, iter_per_epoch * args.warm)
    best_acc = 0
    final_epoch = args.epoch if args.epoch < ITERS/iter_per_epoch else int(ITERS/iter_per_epoch)

    # make dir to store plots, models and logs
    writer = SummaryWriter(log_dir=os.path.join(
            args.log, time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))))
    input_tensor = torch.rand(64, 3, 32, 32).cuda()
    writer.add_graph(model, input_tensor)

    plot_path = os.path.join(PLOT_PATH, time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    model_out_path = os.path.join(args.pth, time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))
    if not os.path.exists(model_out_path):
        os.makedirs(model_out_path)

    for epoch in range(1, final_epoch+1):
        if epoch > args.warm:
            scheduler.step(epoch)
        print("\n===> epoch: {epoch}/{total_epoch}\n".format(epoch=epoch, total_epoch=final_epoch))
        train_loss, train_acc = train()
        test_loss, test_acc = test_training()

        writer.add_scalars('Loss/epoch', {"Train": train_loss, "Test": test_loss}, epoch)
        writer.add_scalars('Accuracy/epoch', {"Train": train_acc, "Test": test_acc}, epoch)

        if epoch >= 15 and test_acc > best_acc:
            save('best_model_with_{acc}_acc'.format(acc=test_acc), model_out_path)
            best_acc = max(best_acc, test_acc)

    conv_visual.visual_filter(model, plot_path)
    conv_visual.visual_feature_map(model, plot_path, device)
    writer.close()
