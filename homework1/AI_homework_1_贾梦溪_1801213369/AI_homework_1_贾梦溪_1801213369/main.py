import argparse
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter
from torchvision.utils import save_image


# #========================================================# #
# #                  1. Model Defination                   # #
# #========================================================# #
class BasicBlock(nn.Module):
    def __init__(self, input, output, stride):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(input, output, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output, output, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output)

        self.Residual = nn.Sequential()
        if stride != 1 or input != output:
            self.Residual = nn.Sequential(
                nn.Conv2d(input, output, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(output)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.Residual(x)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, BasicBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        self.conv2 = nn.Sequential(
            BasicBlock(64, 64, stride=1),
            BasicBlock(64, 64, stride=1)
        )
        self.conv3 = nn.Sequential(
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128, stride=1),
        )
        self.conv4 = nn.Sequential(
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256, stride=1),
        )
        self.conv5 = nn.Sequential(
            BasicBlock(256, 512, stride=2),
            BasicBlock(512, 512, stride=1),
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        conv1 = out
        out, indices = self.max_pooling(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        conv5 = out
        out = F.avg_pool2d(out, out.size(-1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return conv1, conv5, indices, out


def ResNetmodel():
    return ResNet(BasicBlock)


# #========================================================# #
# #                    1.1. Deconv                         # #
# #========================================================# #
class ReverseBasicBlock(nn.Module):
    def __init__(self, input, output, stride):
        super(ReverseBasicBlock, self).__init__()
        if input == output:
            self.conv1 = nn.ConvTranspose2d(output, input, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            self.conv1 = nn.ConvTranspose2d(output, input, kernel_size=3, stride=stride, padding=1, bias=False,
                                            output_padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose2d(output, output, kernel_size=3, stride=1, padding=1, bias=False)

        self.Residual = nn.Sequential()
        if stride != 1 or input != output:
            if input == output:
                self.Residual = nn.Sequential(
                    nn.ConvTranspose2d(output, input, kernel_size=1, stride=stride, bias=False),
                )
            else:
                self.Residual = nn.Sequential(
                    nn.ConvTranspose2d(output, input, kernel_size=1, stride=stride, bias=False, output_padding=1),
                )

    def forward(self, x):

        out = self.conv2(x)
        # out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1(out)
        # out = self.bn2(out)
        out += self.Residual(x)
        out = self.relu(out)

        return out


class ReverseResNet(nn.Module):
    def __init__(self, BasicBlock, indices, num_classes=10, ):
        super(ReverseResNet, self).__init__()
        self.inchannel = 64
        self.indices = indices
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3, bias=False, output_padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.max_unpooling = nn.MaxUnpool2d(kernel_size=3, stride=2, padding=1, )
        self.conv2 = nn.Sequential(

            ReverseBasicBlock(64, 64, stride=1),
            ReverseBasicBlock(64, 64, stride=1),
        )
        self.conv3 = nn.Sequential(
            ReverseBasicBlock(64, 128, stride=2),
            ReverseBasicBlock(128, 128, stride=1),
        )
        self.conv4 = nn.Sequential(
            ReverseBasicBlock(128, 256, stride=2),
            ReverseBasicBlock(256, 256, stride=1),
        )
        self.conv5 = nn.Sequential(
            ReverseBasicBlock(256, 512, stride=2),
            ReverseBasicBlock(512, 512, stride=1),
        )
        self.fc = nn.Linear(512, num_classes)

    def deconv_1(self, x):
        out = self.conv1(x)
        return out

    def deconv_5(self, x):
        out = self.conv5[1](x)
        out = self.conv5[0](out)

        out = self.conv4[1](out)
        out = self.conv4[0](out)

        out = self.conv3[1](out)
        out = self.conv3[0](out)

        out = self.conv2[1](out)
        out = self.conv2[0](out)
        out = self.max_unpooling(out, self.indices, output_size=torch.Size([x.size()[0], 64, 112, 112]))
        out = self.conv1(out)
        return out


def ReverseResNetmodel(indices):
    return ReverseResNet(ReverseBasicBlock, indices)


# #========================================================# #
# #                    2. Solver                           # #
# #========================================================# #
class Solver(object):
    def __init__(self, model, device, criterion):
        super(Solver, self).__init__()
        self.model = model
        self.device = device
        self.criterion = criterion

    def train(self, trainloader, optimizer, writer, epoch):
        self.model.train()
        sum_loss = 0
        total = 0
        correct = 0
        length = len(trainloader)
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            # forward + backward
            _, _, _, outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss = loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            acc = correct / total
            writer.add_scalars('crossentropy_loss', {'crossentropy_loss': loss, }, i + 1 + epoch * length)
            writer.add_scalars('accuracy', {'accuracy': acc}, i + 1 + epoch * length)

            if i % 50 == 49:
                print('Epoch:%d, Iteration:%d, Loss: %.03f , Accuracy: %.3f%% ' % (
                epoch + 1, (i + 1 + epoch * length), loss, 100.0 * correct / total))

    def test(self, testloader, writer, epoch):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                _, _, _, outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        writer.add_scalars('valid_accuracy', {'valid_accuracy': correct / total}, epoch)
        print('The accuracy on valid set: %.3f%%' % (100 * correct / total))


# #========================================================# #
# #                    3. Visualization                    # #
# #========================================================# #
class Visualize(object):
    def __init__(self, ):
        super(Visualize, self).__init__()
        # TO be implemented...

    def denorm(self, tensor, means=(0.4914, 0.4822, 0.4465), stds=(0.2023, 0.1994, 0.2010)):
        for i in range(tensor.size(-1)):
            tensor[:, :, i] = tensor[:, :, i] * stds[i] + means[i]
        return tensor.clamp_(0, 1)

    def plot_filter(self, weights, name="filter", channels_all=True):

        w_min = weights.min().item()
        w_max = weights.max().item()

        channels = [0]
        # make a list of channels if all are plotted
        if channels_all:
            channels = range(weights.shape[1])

        # get number of convolutional filters
        num_filters = weights.shape[0]

        # get number of grid rows and columns
        grid_r, grid_c = 8, 8

        # create figure and axes
        fig, axes = plt.subplots(min([grid_r, grid_c]),
                                 max([grid_r, grid_c]))
        weights = weights.data.cpu()
        # iterate channels
        for channel in channels:
            # iterate filters inside every channel
            for l, ax in enumerate(axes.flat):
                # get a single filter
                img = weights[l, channel, :, :]
                # put it on the grid
                ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
                # remove any labels from the axes
                ax.set_xticks([])
                ax.set_yticks([])
            # save figure
            plt.savefig(osp.join('{}-{}.png'.format(name, channel)), bbox_inches='tight')

    def plot_featmaps(self, conv_img, name):

        w_min = conv_img.min().item()
        w_max = conv_img.max().item()

        # get number of convolutional filters
        num_filters = conv_img.shape[1]

        # get number of grid rows and columns
        grid_r, grid_c = 6, 6

        # create figure and axes
        fig, axes = plt.subplots(min([grid_r, grid_c]),
                                 max([grid_r, grid_c]))

        # iterate filters
        for l, ax in enumerate(axes.flat):
            if name == "images":  # or "reconstruction" in name:
                img = conv_img[l, :, :, :]
                # import pdb;pdb.set_trace()
                img = img.permute(1, 2, 0).data.cpu()
                ax.imshow(self.denorm(img))
                ax.set_xticks([])
                ax.set_yticks([])
                continue;

            f = conv_img[l, :, :, :]

            idx = f.sum(axis=(1, 2)).argmax()
            img = f[idx, :, :].data.cpu()
            # put it on the grid
            ax.imshow(img, interpolation='bicubic')

            # remove any labels from the axes
            ax.set_xticks([])
            ax.set_yticks([])
        # save figure
        plt.savefig(osp.join('{}.png'.format(name)), bbox_inches='tight')

    # #========================================================# #


# #                    4. Main                             # #
# #========================================================# #

def main(args):
    writer = SummaryWriter(args.logs_dir)

    ##==== Data Preprocess ====##
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  #
        transforms.RandomHorizontalFlip(),  #
        transforms.RandomGrayscale(),  #
        transforms.Resize(224, interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224, interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # print(args.data_dir)
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)  #

    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    ##==== Create model ====##
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNetmodel()

    if args.resume:
        print('Resuming checkpoint...\n')
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint, strict=True)

    model = nn.DataParallel(model).to(device)

    ##==== Visualization ====##
    if args.resume:
        vis = Visualize()
        images = next(iter(testloader))[0][:36]
        t_conv1, t_conv5, indices, _ = model(images)
        remodel = ReverseResNetmodel(indices).cuda()
        remodel.load_state_dict(model.module.state_dict(), strict=False)
        vis.plot_filter(model.module.state_dict()["conv1.0.weight"])
        vis.plot_featmaps(images, name="images")
        vis.plot_featmaps(t_conv1, name="conv1-featuremap")
        vis.plot_featmaps(remodel.deconv_1(t_conv1), name="conv1-reconstruction")
        vis.plot_featmaps(t_conv5, name="conv5-featuremap")
        vis.plot_featmaps(remodel.deconv_5(t_conv5), name="conv5-reconstruction")

        return

    ##==== Criterion ====##
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    def adjust_lr(epoch):
        lr = args.lr if epoch <= 10 else \
            args.lr * (0.01 ** ((epoch - 10) / 90))
        for g in optimizer.param_groups:
            g['lr'] = lr

    solver = Solver(model, device, criterion)
    ##==== Train ====##
    for epoch in range(args.epoches):
        adjust_lr(epoch)
        print('Epoch: {}'.format(epoch + 1))
        # train
        solver.train(trainloader, optimizer, writer, epoch)
        # test
        solver.test(testloader, writer, epoch)
        if epoch % 10 == 9:
            torch.save(model.module.state_dict(), "checkpoint.pth.tar")

    torch.save(model.module.state_dict(), "final-checkpoint.pth.tar")

    vis = Visualize()
    images = next(iter(testloader))[0][:36]
    t_conv1, t_conv5, indices, _ = model(images)
    remodel = ReverseResNetmodel(indices).cuda()
    remodel.load_state_dict(model.module.state_dict(), strict=False)

    vis.plot_featmaps(images, name="images")
    vis.plot_featmaps(t_conv1, name="conv1-featuremap")
    vis.plot_featmaps(remodel.deconv_1(t_conv1), name="conv1-reconstruction")
    vis.plot_featmaps(t_conv5, name="conv5-featuremap")
    vis.plot_featmaps(remodel.deconv_5(t_conv5), name="conv5-reconstruction")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AI homework: mxjia")
    # data
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        default='../data')
    parser.add_argument('--logs_dir', type=str, metavar='PATH',
                        default="logs")
    parser.add_argument('--resume', type=str, metavar='PATH',
                        default='')
    parser.add_argument('--batch_size', type=int, default=64, )
    parser.add_argument('--epoches', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.1)
    main(parser.parse_args())