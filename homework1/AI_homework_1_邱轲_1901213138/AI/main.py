import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import numpy as np
from tensorboardX import SummaryWriter
from Resnet18 import Resnet_18
from Dataloader import Load_Cifar10
from Visualization import Filter_visualization, Featuremap_visualization
import warnings
warnings.filterwarnings("ignore")

# Parameters setting
parser = argparse.ArgumentParser(description='Train/Test Resnet18 based on Cifar-10 with Pytorch')
parser.add_argument('--gpu',                type=bool,   default=torch.cuda.is_available(),    help='gpu or cpu')
parser.add_argument('--lr',                 type=float,  default=0.001,                        help='Initial learning rate')
parser.add_argument('--batch_size',         type=int,    default=100,                          help='batch size for train and test')
parser.add_argument('--num_epoch',          type=int,    default=20,                           help='Epoch Times of training')
parser.add_argument('--checkpoint_path',    type=str,    default='./check_point/',             help='Path to save model')
parser.add_argument('--log_path',           type=str,    default='./qiuke/',                   help='Path to save tensorboardX')
parser.add_argument('--visual_path',        type=str,    default='./Visualization/',           help='Path to save Filter/Featuremap Visualization')

args = parser.parse_args()

# Essential information of Cifar10
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def train(epoch):

    model.train()
    sum_loss = 0.0
    correct_num = 0
    total_num = 0
    acc = 0

    for batch_num, (train_data, label) in enumerate(train_loader):
        train_data, label = train_data.to(device), label.to(device)
        optimizer.zero_grad()

        # run a batch
        result = model(train_data)
        loss = loss_function(result, label)
        loss.backward()
        optimizer.step()

        # calculate Acc and Loss
        sum_loss += loss.item()
        # print(result)
        _, predicted = torch.max(result, 1)
        # print(predicted)
        total_num += label.size(0)
        # print(label.size(0))
        correct_num += np.sum(predicted.cpu().numpy() == label.cpu().numpy())
        acc = 100. * correct_num/total_num

        # print for each batch
        step_in_a_epoch = len(train_loader)
        trained_step = 1 + step_in_a_epoch * epoch + batch_num
        total_step = step_in_a_epoch * args.num_epoch
        trained_num = batch_num * args.batch_size + len(train_data)
        # total_sam = len(train_loader.dataset)

        # loss.item Correspond to current batch loss;
        # sum_loss/(batch_num+1)  Correspond to the average batch loss in current epoch
        print('Epoch:{}/{}  Step:{}/{}  Trained_num:{}/{}  Train_Loss:{:0.4f}  Avg_loss:{:0.4f}  lr:{:0.5f}  Acc:{:0.3f}%'.format(
            epoch+1, args.num_epoch,
            trained_step, total_step,
            trained_num, len(train_loader.dataset),
            loss.item(), sum_loss/(batch_num+1),
            optimizer.param_groups[0]['lr'], acc))

        # record for tensorboard
        #writer.add_scalar('Train_loss', loss.item(), global_step=trained_step)
        writer.add_scalar('Train_Accuracy', acc, global_step=trained_step)
        writer.add_scalars('Train_loss', {'Train_Loss': loss.item(), 'Avg_loss': sum_loss/(batch_num+1)}, global_step=trained_step)
    return step_in_a_epoch, sum_loss/step_in_a_epoch, acc

def Test(epoch):
    # BN won't be changed
    model.eval()
    sum_loss = 0.0
    correct_num = 0
    total_num = 0
    # don't calculate grad
    with torch.no_grad():
        # chose one batch of
        for batch_num, (test_data, label) in enumerate(test_loader):
            test_data, label = test_data.to(device), label.to(device)
            result = model(test_data)
            loss = loss_function(result, label)

            # calculate Acc and Loss
            sum_loss += loss.item()
            _, predicted = result.max(1)
            total_num += label.size(0)
            correct_num += np.sum(predicted.cpu().numpy() == label.cpu().numpy())

        test_acc = 100. * correct_num/total_num
        test_loss = sum_loss/len(test_loader)
        print('Epoch:{epoch}/{total_epoch}  Test_Loss:{:0.4f}  Test_Acc:{:0.4f}%'.format(
            test_loss, test_acc, epoch=epoch+1, total_epoch=args.num_epoch))

    # record for tensorboard
    writer.add_scalar('Test_loss', test_loss, epoch)
    writer.add_scalar('Test_Accuracy', test_acc, epoch)

    return test_loss, test_acc


if __name__ == '__main__':

    # chose gpu
    if args.gpu:
        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    # Load_Cifar10()
    train_loader, test_loader = Load_Cifar10(args.batch_size)

    # Load Resnet_18
    model = Resnet_18().to(device)

    # define lr,loss_f, optimizer
    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15, 18], gamma=0.4)
    # warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    # prepare for tensorboard
    writer = SummaryWriter(log_dir=args.log_path)

    # Train and Test
    print("start train!")
    best_Acc = 80
    Train_Loss = 0.0
    Train_Acc = 0.0
    Test_Loss = 0.0
    Test_Acc = 0.0
    for epoch in range(args.num_epoch):
        # Train/Test per epoch
        step_per_epoch, Train_Loss, Train_Acc = train(epoch)
        scheduler.step(epoch)
        Test_Loss, Test_Acc = Test(epoch)

        # record for tensorboardX
        writer.add_scalars('Contrast on Loss', {'Train_Loss': Train_Loss, 'Test_Loss': Test_Loss}, epoch)
        writer.add_scalars('Contrast on Acurracy', {'Train_Acc': Train_Acc, 'Test_Acc': Test_Acc}, epoch)

        # check checkpoint
        if not os.path.exists(args.checkpoint_path):
            os.makedirs(args.checkpoint_path)

        # chose model to save
        current_step = step_per_epoch * (epoch + 1)
        if (epoch + 1) == args.num_epoch:
            torch.save(model, args.checkpoint_path + 'Resnet18_step{step}.pth'.format(step=current_step))

        if Test_Acc > best_Acc and epoch > 0.75 * args.num_epoch:
            best_Acc = Test_Acc
            torch.save(model, args.checkpoint_path + 'Resnet18_step{step}_better.pth'.format(step=current_step))


    # Filter visualization based on Plot and Torchvision
    Filter_visualization(model, args.log_path, args.visual_path, current_step)

    # Featuremap visualization based on Reverse_Resnet18
    Featuremap_visualization(model, args.visual_path)
    print('Featuremap Visualization Succeeded!')
    print('done! Best accuracy = {:.2f}%'.format(best_Acc))
    print("Hi~ Here are some warnings that don't affect the results\n Because I use PLT to draw single channel images.")

    writer.export_scalars_to_json("./tensorboardX.json")
    writer.close()







