'''
zhuliwen: liwenzhu@pku.edu.cn
October 24，2019
ref: https://github.com/weiaicunzai/pytorch-cifar100
'''

from AI_homework_1 import *
import torch

def test(pth_file):
    net = ResNet(BasicBlock, [2, 2, 2, 2])
    net.load_state_dict(torch.load(pth_file))
    net.cuda()
    print(net)
    net.eval()
    correct_1 = 0.0
    correct_5 = 0.0
    test_loss = 0.0

    for n_iter, (image, label) in enumerate(cifar10_test_loader):
        print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar10_test_loader)))
        image = Variable(image).cuda()
        label = Variable(label).cuda()
        output = net(image)
        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(output, label)
        test_loss += loss.item()

        _, pred = output.topk(5, 1, largest=True, sorted=True)

        label = label.view(label.size(0), -1).expand_as(pred)
        correct = pred.eq(label).float()

        # compute top 5
        correct_5 += correct[:, :5].sum()

        # compute top1
        correct_1 += correct[:, :1].sum()

    print()
    print("Top 1 Accuracy: ", correct_1 / len(cifar10_test_loader.dataset))
    print("Top 5 Accuracy: ", correct_5 / len(cifar10_test_loader.dataset))
    print("Average Loss:", test_loss / len(cifar10_test_loader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))


if __name__ == '__main__':
    cifar10_test_loader = get_test_dataloader(
        settings.CIFAR10_TEST_MEAN,
        settings.CIFAR10_TEST_STD,
        num_workers=2,
        batch_size=128,
        shuffle=True
    )

    pth_file = 'resnet18-25-best.pth'
    test(pth_file)

