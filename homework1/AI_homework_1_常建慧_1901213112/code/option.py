import argparse
import torch


def gather_options():
    parser = argparse.ArgumentParser(description='Resnet settings')

    parser.add_argument('--niter', type=int, default='25', help="iter of starting learning decay")
    parser.add_argument('--niter_decay', type=int, default='25', help="iter of decay learning rate to zero")
    parser.add_argument('--epoch_count', type=int, default='1', help='number of the start epoch')

    parser.add_argument('--lr', type=float, default=0.01, help="inital learning rate for adam")
    parser.add_argument('--lr_policy', type=str, default='step', help='earning rate policy: lambda|step|plateau')
    parser.add_argument('--lr_decay_iters', type=int, default=20,
                        help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--dataroot', type=str, default='./data/',
                        help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
    parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
    parser.add_argument('--num_classes', type=int, default=10, help='# of output image channels')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2, -1 for CPU mode')

    parser.add_argument('--name', type=str, default='resnet',
                        help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--epoch', type=str, default='latest',
                        help='which epoch to load? set to latest to use latest cached model')

    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--nl', type=str, default='relu', help='non-linearity activation: relu | lrelu | elu')
    parser.add_argument('--norm', type=str, default='batch', help='instance normalization or batch normalization')

    parser.add_argument('--display_port', type=int, default=8097, help='visdom display port')
    parser.add_argument('--display_server', type=str, default="http://localhost",
                        help='visdom server of the web display')
    parser.add_argument('--print_freq', type=int, default='300', help='print loss values every print_freq epoch')
    parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
    parser.add_argument('--save_epoch_freq', type=int, default=5,
                        help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
    parser.add_argument('--num_test', type=int, default=20, help="num of test imgaes")

    opt = parser.parse_args()
    # if len(opt.gpu_ids) > 0:
    #     torch.cuda.set_device(int(opt.gpu_ids[0]))
    return opt


def print_options(opt):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)
