import os

import torch


def save_networks(opt, net, epoch):
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    save_filename = '%s_net_resnet.pth' % epoch
    save_path = os.path.join(save_dir, save_filename)

    if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
        torch.save(net.cpu().state_dict(), save_path)
        net.cuda()
    else:
        torch.save(net.cpu().state_dict(), save_path)


def load_networks(opt, net):
    load_filename = '%s_net_resnet.pth' % opt.epoch
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    load_path = os.path.join(save_dir, load_filename)
    if isinstance(net, torch.nn.DataParallel):
        net = net.module
    print('loading the model from %s' % load_path)
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
    state_dict = torch.load(load_path, map_location=device)
    net.load_state_dict(state_dict)
    # print(net)


def evaluate(net, testloader, device):
    total = 0
    correct = 0
    for i, data in enumerate(testloader):
        inputs = data[0].to(device)
        labels = data[1].to(device)
        total += labels.size(0)
        outputs = net(inputs)
        _, predict = torch.max(outputs, 1)
        correct += (predict == labels).sum().item()
    return float(correct * 100 / total)
