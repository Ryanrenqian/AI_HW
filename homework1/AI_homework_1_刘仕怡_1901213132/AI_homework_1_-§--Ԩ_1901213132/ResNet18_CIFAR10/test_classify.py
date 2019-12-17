# coding with UTF-8
# ******************************************
# *****CIFAR-10 with ResNet8 in Pytorch*****
# *****test_classify.py                *****
# *****Author：Shiyi Liu               *****
# *****Time：  Oct 22nd, 2019          *****
# ******************************************import torch
import torch
import torch.backends.cudnn as cudnn
import conv_visual
import train_classify
import argparse
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

PTH = r'model/2019-10-25-00-25-26/best_model_with_0.9203_acc.pth'
VPATH = 'plots/test/'


def load_model_from_path(pthfile):
    return torch.load(pthfile)


def test():
    model.eval()
    batch_loss = 0.0
    acc = 0.0
    total = 0.0
    with torch.no_grad():
        for batch_num, (img, label) in enumerate(test_loader):
            img, label = img.to(device), label.to(device)

            output = model(img)
            loss = loss_function(output, label)
            batch_loss += loss.item()
            outlabel = torch.max(output, 1)
            total += label.size(0)
            acc += np.sum(outlabel[1].cpu().numpy() == label.cpu().numpy())

        print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}% ({acc}/{total})'.format(
            batch_loss / len(test_loader),
            100.0 * acc / total,
            acc=int(acc),
            total=int(total)
        ))

    return batch_loss / len(test_loader.dataset), acc/total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="cifar-10 with ResNet18 in PyTorch")
    parser.add_argument('--visnum', default=100, type=int, help='visual num')
    parser.add_argument('--testBatchSize', default=100, type=int, help='testing batch size')
    parser.add_argument('--pth', default=PTH, help='training batch size')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    args = parser.parse_args()
    pthfile = args.pth

    device = train_classify.get_device()
    model = load_model_from_path(pthfile).to(device)
    mean, std = train_classify.load_mestd()
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    test_loader = DataLoader(dataset=test_set, batch_size=16)

    loss_function = torch.nn.CrossEntropyLoss().to(device)

    test()

    conv_visual.visual_filter(model, VPATH)
    conv_visual.visual_feature_map(model, VPATH, device)



