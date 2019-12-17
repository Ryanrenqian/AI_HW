import argparse
import os
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import model_1 as RN

from tensorboardX import SummaryWriter
import utility
import scipy.misc as misc

from tensorboard_logger import configure, log_value

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--net_type', default='resnet', type=str,
                    help='networktype: resnet, resnext, densenet, pyamidnet, and so on')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='check_point/hw_yd/checkpoint.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',default=True,
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model on ImageNet-1k dataset')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--depth', default=32, type=int,
                    help='depth of the network (default: 32)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='to use basicblock for CIFAR datasets (default: bottleneck)')
parser.add_argument('--dataset', dest='dataset',type=str,default='cifar10',
                    help='dataset (options: cifar10, cifar100, and imagenet)')
parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                    help='to print the status at every iteration')
parser.add_argument('--alpha', type=int,
                    help='number of new channel increases per depth (default: 300)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation for CIFAR datasets (default: True)')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--expname',  type=str,
                    help='name of experiment')

parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=True)
parser.set_defaults(augment=True)

best_err1 = 0
best_err5 = 0


def save_results_test(self, filename, save_list, scale):

    filename = '{}/results/{}/N{}/{}'.format(self.dir, self.args.testset, str(self.args.noise_level[0]), filename)
    postfix = ('DN', 'LQ', 'HQ')
    for v, p in zip(save_list, postfix):
        normalized = v[0].data.mul(255 / self.args.rgb_range)
        ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
        misc.imsave('{}.png'.format(filename), ndarr)
def main():
    global args, best_err1, best_err5
    args = parser.parse_args()
    if args.tensorboard: configure("runs/%s"%(args.expname))

    args.distributed = args.world_size > 1
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    if args.dataset.startswith('cifar'):
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        if args.augment:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                normalize,
                ])
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize
            ])

        if args.dataset == 'cifar10':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('data', train=True, download=True, transform=transform_train),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('data', train=False, transform=transform_test),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            numberofclass = 10
        else:
            raise Exception ('unknown dataset: {}'.format(args.dataset))


    else:
        raise Exception ('unknown dataset: {}'.format(args.dataset))

    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.net_type))
        try:
            model = models.__dict__[str(args.net_type)](pretrained=True)
        except (KeyError, TypeError):
            print('unknown model')
            print('torchvision provides the follwoing pretrained model:', model_names)
            return
    else:
        print("=> creating model '{}'".format(args.net_type))
        if args.net_type == 'resnet':
            model = RN.youdi() # for ResNet
            dijiceng=1
            for m in model.modules():
                if isinstance(m, nn.Conv2d):

                    if dijiceng==1:
                        abc = m.weight.data
                        d=m.weight
                        # print(abc)

                    dijiceng+=1
        # abc= utility.quantize(abc, 1)
        # for i in range(64):
        #     wei=abc[i,:,:,:]
        #
        #     # range=torch.max(wei)-torch.min(wei)
        #     # print(torch.max(wei))
        #     # max=torch.max(wei).cpu().numpy()
        #     # min=torch.min(wei).cpu().numpy()
        #
        #
        #     normalized = wei.data.permute(1, 2, 0).cpu().numpy()
        #     normalized = normalized* 255
        #     max=np.max(normalized)
        #     min=np.min(normalized)
        #     normalized_add=normalized-min
        #     ndarr=normalized_add/(max-min)
        #     # ndarr=normalized+min
        #     # normalized = wei.data.mul(255 / 1)
        #     # ndarr = normalized.byte().cpu().numpy()
        #
        #     # ndarr=ndarr/range
        #     ndarr=ndarr*255
        #     misc.imsave('{}.png'.format(i), ndarr)

        im, label = next(iter(val_loader))
        convt_1 = nn.ConvTranspose2d(64, 3, 7, stride=2, padding=3, bias=False)
        convt_1.weight = model.conv1.weight
        im1 = im[0]
        normalized = im1.data.permute(1, 2, 0).cpu().numpy()
        # normalized = normalized * 255
        # max = np.max(normalized)
        # min = np.min(normalized)
        # normalized_add = normalized - min
        # ndarr = normalized_add / (max - min)
        # # ndarr=normalized+min
        # # normalized = wei.data.mul(255 / 1)
        # # ndarr = normalized.byte().cpu().numpy()
        #
        # # ndarr=ndarr/range
        # ndarr = ndarr * 255
        misc.imsave('pic_2.png', normalized)

        out=model.conv1(im)
        # out=out[0].squeeze(0)
        out[0,1:64]=0

        out=convt_1(out)
        out = out[0]
        # out=out.squeeze(0)
        normalized1 = out.data.permute(1, 2, 0).cpu().numpy()
        misc.imsave('pic_feature_2.png', normalized1)















    if not args.distributed:
        if args.net_type.startswith('alexnet') or args.net_type.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    print(model)
    print('the number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    #
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_err1 = checkpoint['best_acc1']
            best_err5 = checkpoint['best_acc5']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.evaluate:
        err1,err5=validate(val_loader, model, criterion,80)
        print(err1,err5)
        return
    writer = SummaryWriter()




def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    current_LR = get_learning_rate(optimizer)[0]
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))

        top5.update(err5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: {LR:.6f}\t' 
                  'batch_time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'data_time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})'.format(
                   epoch, args.epochs, i, len(train_loader), LR=current_LR, batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
    return losses.avg,top1.avg


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        print(input)
        print(target)


        target = target.cuda(async=True)


        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)


            output = model(input_var)

            loss = criterion(output, target_var)


        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Test (on val set): [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                   epoch, args.epochs, i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'.format(epoch, args.epochs, top1=top1, top5=top5, loss=losses))
    return top1.avg, top5.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = "check_point/%s/"%(args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'check_point/%s/'%(args.expname) + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.dataset.startswith('cifar'):
        lr = args.lr * (0.1 ** (epoch // (args.epochs*0.5))) * (0.1 ** (epoch // (args.epochs*0.75)))
    elif args.dataset == ('imagenet'):
        lr = args.lr * (0.1 ** (epoch // 30))

    if args.tensorboard:
        log_value('learning_rate', lr, epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]
    return lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
