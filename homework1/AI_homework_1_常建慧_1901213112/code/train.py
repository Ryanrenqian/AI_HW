from option import gather_options, print_options
from network import Resnet, get_scheduler, init_net
from dataload import loadData
from Util import save_networks, load_networks, evaluate
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision

if __name__ == '__main__':
    opt = gather_options()
    print_options(opt)

    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

    trainloader, testloader = loadData(opt)
    dataset_size = len(trainloader)
    print('#training images = %d' % dataset_size)

    net = Resnet(opt.input_nc, num_classes=opt.num_classes, norm=opt.norm, nl=opt.nl)
    net = init_net(net, init_type='normal', gpu_ids=[0])

    if opt.continue_train:
        load_networks(opt, net)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9)
    scheduler = get_scheduler(optimizer, opt)

    iter = 0
    running_loss = 0.0
    correct = 0.0
    total = 0

    writer = SummaryWriter()
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        loss = 0.0
        for i, data in enumerate(trainloader):
            iter = iter + 1
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total += labels.size(0)
            _, predict = torch.max(outputs.data, 1)
            correct += (predict == labels).sum().item()
            if iter % opt.print_freq == 0:
                writer.add_scalar('Loss_crossEntropy/train', float(running_loss / opt.print_freq), iter)
                # trainset accuracy
                accuracy = correct * 100.0 / total
                writer.add_scalar('Accuracy/train', accuracy, iter)
                print("iteration: %d, loss: %.4f, accuracy on %d train images: %.3f %%"
                      % (iter, running_loss / opt.print_freq, total, accuracy))
                writer.add_graph(net, inputs)
                running_loss = 0.0
                correct = 0
                total = 0
            if iter % opt.save_latest_freq == 0:
                save_networks(opt, net, 'latest')
                print('saving the latest model (epoch %d, iter %d)' % (epoch, iter))

        # testset accuracy
        test_accuracy = evaluate(net, testloader, device)
        print("Accuracy on testset of epoch %d (iter: %d )is %.3f %%" % (epoch, iter, test_accuracy))
        writer.add_scalar('Accuracy/test', test_accuracy, iter)

        if epoch % opt.save_epoch_freq == 0:
            save_networks(opt, net, epoch)

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
    writer.close()
