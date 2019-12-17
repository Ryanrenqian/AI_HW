"""Train CIFAR10 with PyTorch."""
import argparse
import math
import os
import sys
import time
from bisect import bisect_right
import heapq
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import tqdm
from torch.utils.tensorboard import SummaryWriter


class MinHeap(object):
    def __init__(self, key):
        self.key = key
        self.heap = []

    def push(self, item):
        decorated = self.key(item), item
        heapq.heappush(self.heap, decorated)

    def pop(self):
        key, item = heapq.heappop(self.heap)
        return item

    def pushpop(self, item):
        decorated = self.key(item), item
        rkey, ritem = heapq.heappushpop(self.heap, decorated)
        return ritem

    def __len__(self):
        return len(self.heap)

    def __getitem__(self, index):
        return self.heap[index][1]


class Checkpointer(object):
    def __init__(self, save_dir, keep_best=5, keep_last=5):
        self.save_dir = save_dir
        self.keep_best = keep_best
        self.keep_last = keep_last
        self.checkpoints = MinHeap(lambda x: x[0])

        self.ckpt_pattern = os.path.join(self.save_dir, "checkpoint_{}.pt")
        self.glob_pattern = os.path.join(self.save_dir, "checkpoint_*.pt")
        self.meta_path = os.path.join(self.save_dir, "checkpoint_meta.txt")

        self._read_meta()

    def _read_meta(self):
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf8") as f:
                for line in f:
                    accuracy, ckpt = line.split()
                    accuracy = float(accuracy)
                    if len(self.checkpoints) < self.keep_best:
                        self.checkpoints.push((accuracy, ckpt))
                    else:
                        self.checkpoints.pushpop((accuracy, ckpt))

    def _write_meta(self):
        with open(self.meta_path, "w", encoding="utf8") as f:
            for acc, ckpt in self.checkpoints:
                f.write(f"{acc}\t{ckpt}\n")
                f.flush()

    def _cleanup(self, ckpts_to_keep):
        ckpts = glob.glob(self.glob_pattern)
        for ckpt in ckpts:
            if ckpt not in ckpts_to_keep:
                os.remove(ckpt)
                print(f"checkpoint {ckpt} is removed")

    def save(self, last_epoch, data, accuracy=None):
        start = time.time()
        ckpt_name = self.ckpt_pattern.format(last_epoch)
        torch.save(data, ckpt_name)
        print(
            f"checkpoint saved to {ckpt_name} (used {time.time() - start:.3f}s)"
        )

        if (
            accuracy is not None
            and self.keep_best is not None
            and self.keep_best > 0
        ):
            if len(self.checkpoints) < self.keep_best:
                self.checkpoints.push((accuracy, ckpt_name))
            else:
                self.checkpoints.pushpop((accuracy, ckpt_name))
            best_checkpoints = [x[1] for x in self.checkpoints]
        else:
            best_checkpoints = []

        if self.keep_last is not None and self.keep_last > 0:
            last_checkpoints = [
                self.ckpt_pattern.format(last_epoch - x)
                for x in range(self.keep_last)
            ]
        else:
            last_checkpoints = []

        if best_checkpoints or last_checkpoints:
            checkpoints = set(best_checkpoints + last_checkpoints)
            self._cleanup(checkpoints)

        if self.checkpoints:
            self._write_meta()

    def load(self, epoch):
        if epoch >= 0:
            ckpt_name = self.ckpt_pattern.format(epoch)
            if os.path.exists(ckpt_name):
                state_dict = torch.load(ckpt_name, "cpu")
                print(f"checkpoint {ckpt_name} loaded")
                return (
                    state_dict["net"],
                    state_dict["optimizer"],
                    state_dict["last_epoch"],
                )
            print(f"checkpoint {ckpt_name} does not exist")
        return None, None, None


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10/100 Training")
    parser.add_argument("save_dir", type=str, help="save dir")
    parser.add_argument(
        "--model",
        default="resnet18",
        type=str,
        help="model",
        choices=[
            "resnet",
            "net",
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
        ],
    )
    parser.add_argument("--num_label", default=10, type=int, choices=[10, 100])
    parser.add_argument(
        "--scheduler",
        default="normal",
        choices=["normal", "renorm", "renorm2", "renorm3", "custom"],
        type=str,
        help="scheduler",
    )
    parser.add_argument("--scheduler_steps", type=int, nargs="*")
    parser.add_argument("--num_blocks", default=3, type=int)
    parser.add_argument(
        "--optim",
        default="sgd",
        type=str,
        help="optimizer",
        choices=["sgd", "adam"],
    )
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--lrs", type=float, nargs="*")
    parser.add_argument("--gamma", default=0.1, type=float, help="gamma")
    parser.add_argument(
        "--momentum", default=0.9, type=float, help="momentum term"
    )
    parser.add_argument(
        "--restore", default=-1, type=int, help="restore training"
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0002,
        type=float,
        help="weight decay for optimizers",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--renorm", action="store_true")
    parser.add_argument(
        "--renorm_mode", type=str, default="C", choices=["C", "CHW", "CHW/C"]
    )
    args = parser.parse_args()
    if not args.save_dir.startswith("save"):
        args.save_dir = os.path.join("save", args.save_dir)
    print(args)

    try:
        os.makedirs(args.save_dir)
    except OSError:
        if args.restore < 0:
            print(f"save_dir {args.save_dir} already exists, please check")
            sys.exit()

    return args


def build_dataset(args):
    # print("==> Preparing data..")
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    if args.num_label == 10:
        trainset = torchvision.datasets.CIFAR10(
            root="data", train=True, download=True, transform=transform_train
        )
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=32, shuffle=True
        )

        testset = torchvision.datasets.CIFAR10(
            root="data", train=False, download=True, transform=transform_test
        )
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=250, shuffle=False
        )
        # classes = ('plane', 'car', 'bird', 'cat', 'deer',
        #     'dog', 'frog', 'horse', 'ship', 'truck')
    else:
        trainset = torchvision.datasets.CIFAR100(
            root="data", train=True, download=True, transform=transform_train
        )
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True
        )

        testset = torchvision.datasets.CIFAR100(
            root="data", train=False, download=True, transform=transform_test
        )
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=250, shuffle=False
        )

    return train_loader, test_loader


def get_conv_layer(
    input_channel, output_channel, kernel_size, stride, norm="none", bias=False
):
    if (kernel_size - 1) % 2 != 0:
        raise ValueError("only symmetric padding")
    padding = (kernel_size - 1) // 2

    if norm == "none":
        return nn.Conv2d(
            input_channel,
            output_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
    elif norm == "pre":
        bn = nn.BatchNorm2d(input_channel, eps=0.001)
        relu = nn.ReLU()
        conv = nn.Conv2d(
            input_channel,
            output_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        return nn.Sequential(*[bn, relu, conv])
    elif norm == "post":
        conv = nn.Conv2d(
            input_channel,
            output_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        bn = nn.BatchNorm2d(output_channel, eps=0.001)
        relu = nn.ReLU()
        return nn.Sequential(*[conv, bn, relu])
    else:
        raise NotImplementedError("Unknown Normaliztion Method")


class LayerNormNCHW(nn.Module):
    def __init__(self, normalized_shape, eps=1e-05, affine=True):
        super(LayerNormNCHW, self).__init__()
        self.ln = nn.LayerNorm(
            normalized_shape, eps=eps, elementwise_affine=affine
        )

    def forward(self, x):
        # x is (N, C, H, W)
        # reshape to (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        # reshape back to (N, C, H, W)
        x = x.permute(0, 3, 1, 2)
        return x


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape=None, eps=1e-05, affine_shape=None):
        assert normalized_shape is not None or affine_shape is not None
        super(LayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.affine_shape = affine_shape

        if self.normalized_shape is not None:
            self.normalize = True
            self.normalized_dims = tuple(
                -1 - x
                for x, v in enumerate(reversed(normalized_shape))
                if v > 0
            )
        else:
            self.normalize = False

        if self.affine_shape:
            self.affine = True
            self.weight = nn.Parameter(torch.ones(*affine_shape))
            self.bias = nn.Parameter(torch.zeros(*affine_shape))
        else:
            self.affine = False
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.normalize:
            std, mean = torch.std_mean(x, self.normalized_dims, True, True)
            x = (x - mean) / (std + self.eps)

        if self.affine:
            x = self.weight * x + self.bias
        return x

    def extra_repr(self):
        return "normalized_shape={normalized_shape}, eps={eps}, affine_shape={affine_shape}".format(
            **self.__dict__
        )


class ZeroPad(nn.Module):
    def __init__(self, dim, left_pad, right_pad):
        super(ZeroPad, self).__init__()
        self.dim = dim
        self.left_pad = left_pad
        self.right_pad = right_pad

    def forward(self, x):
        x = torch.transpose(x, self.dim, -1)
        x = F.pad(x, (self.left_pad, self.right_pad), "constant", 0)
        x = torch.transpose(x, self.dim, -1)
        return x


class Block(nn.Module):
    def __init__(
        self,
        input_channel,
        output_channel,
        kernel_size=3,
        is_first=False,
        norm_first=False,
        downsample="pad",
        renorm_shape=None,
        affine_shape=None,
    ):
        super(Block, self).__init__()
        # in the 6n+2 structure
        # channel number doubles: 16 -> 32 -> 64
        # feature map size halves: 32*32 -> 16*16 -> 8*8
        if input_channel == output_channel:
            stride = 1
            self.downsample = None
        elif (
            output_channel % input_channel == 0
            and output_channel // input_channel == 2
        ):
            stride = 2
            if downsample == "pad":
                avg_pool = nn.AvgPool2d(stride, stride)
                zero_pad = ZeroPad(1, input_channel // 2, input_channel // 2)
                self.downsample = nn.Sequential(avg_pool, zero_pad)
            elif downsample == "conv":
                if is_first and norm_first:
                    conv = get_conv_layer(
                        input_channel, output_channel, 1, stride, "none"
                    )
                else:
                    conv = get_conv_layer(
                        input_channel,
                        output_channel,
                        1,
                        stride,
                        "pre" if norm_first else "post",
                    )
                self.downsample = conv
            else:
                raise NotImplementedError("Unknown downsample method")
        else:
            raise NotImplementedError("Unknown block expansion")

        if is_first and norm_first:
            self.conv1 = get_conv_layer(
                input_channel, output_channel, kernel_size, stride, "none"
            )
        else:
            self.conv1 = get_conv_layer(
                input_channel,
                output_channel,
                kernel_size,
                stride,
                "pre" if norm_first else "post",
            )

        self.conv2 = get_conv_layer(
            output_channel,
            output_channel,
            kernel_size,
            1,
            "pre" if norm_first else "post",
        )

        if renorm_shape is not None:
            self.layernorm1 = LayerNorm(
                renorm_shape, eps=1e-06, affine_shape=affine_shape
            )
            self.layernorm2 = LayerNorm(
                renorm_shape, eps=1e-06, affine_shape=affine_shape
            )
        else:
            self.layernorm1 = None
            self.layernorm2 = None

        self.reset_parameters()

    def reset_parameters(self):
        # downsample could be None, Seq[AvgPool, ZeroPad], Conv2d, Seq{ReLU, BN, Conv2d}
        # conv1 could be Conv2d, Seq{ReLU, BN, Conv2d}
        # conv2 is Seq{ReLU, BN, Conv2d}
        # layernorm{1,2} could be None, LN
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Sequential):
                for n in m:
                    if isinstance(n, nn.Conv2d):
                        nn.init.xavier_uniform_(n.weight)
                    elif isinstance(n, nn.BatchNorm2d):
                        nn.init.zeros_(n.bias)
                        nn.init.ones_(n.weight)
            if isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x = x + shortcut
        if self.layernorm1 is not None:
            x = self.layernorm1(x)
            x = x + shortcut
            x = self.layernorm2(x)
        return x


class Net(nn.Module):
    def __init__(
        self,
        num_blocks,
        num_classes=10,
        kernel_size=3,
        norm_first=False,
        downsample="pad",
        renorm=False,
    ):
        super(Net, self).__init__()

        self.kernel_size = kernel_size
        self.norm_first = norm_first
        self.downsample = downsample

        if renorm:
            renorm_shapes = [(16, 32, 32), (32, 16, 16), (64, 8, 8)]
        else:
            renorm_shapes = [None, None, None]

        self.num_blocks = num_blocks
        self.conv1 = get_conv_layer(3, 16, 3, 1, "post")

        self.layer1 = self._make_layer(16, 16, True, renorm_shapes[0])
        self.layer2 = self._make_layer(16, 32, False, renorm_shapes[1])
        self.layer3 = self._make_layer(32, 64, False, renorm_shapes[2])

        self.linear = nn.Linear(64, num_classes)
        if self.norm_first:
            self.linear_pre_norm = nn.Sequential(
                nn.BatchNorm2d(64, eps=0.001), nn.ReLU()
            )
        else:
            self.linear_pre_norm = None

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
                # uniform_unit_scaling_initializer
                # [-sqrt(3)/sqrt(size of input), sqrt(3)/sqrt(size of input)]
                scale = math.sqrt(3 / m.in_features)
                nn.init.uniform_(m.weight, -scale, scale)
            elif isinstance(m, nn.Sequential):
                for n in m:
                    if isinstance(n, nn.Conv2d):
                        nn.init.xavier_uniform_(n.weight)
                    elif isinstance(n, nn.BatchNorm2d):
                        nn.init.zeros_(n.bias)
                        nn.init.ones_(n.weight)

    def _make_layer(
        self, input_channel, output_channel, is_first=False, renorm_shape=None
    ):
        input_channels = [input_channel] + [output_channel] * (
            self.num_blocks - 1
        )

        layers = []
        for idx, input_channel in enumerate(input_channels):
            if is_first and idx == 0:
                layer = Block(
                    input_channel,
                    output_channel,
                    self.kernel_size,
                    True,
                    self.norm_first,
                    self.downsample,
                    renorm_shape,
                )
            else:
                layer = Block(
                    input_channel,
                    output_channel,
                    self.kernel_size,
                    False,
                    self.norm_first,
                    self.downsample,
                    renorm_shape,
                )
            layers.append(layer)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if self.linear_pre_norm is not None:
            x = self.linear_pre_norm(x)

        # 8x8 avg pool
        x = torch.mean(x, (2, 3))
        x = self.linear(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, in_planes, planes, stride=1, renorm_shape=None, affine_shape=None
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )
        if renorm_shape is not None or affine_shape is not None:
            if (
                renorm_shape[1] == 0
                and renorm_shape[2] == 0
                and affine_shape[1] == 1
                and affine_shape[2] == 1
            ):
                self.layer_norm1 = LayerNormNCHW(renorm_shape[0], affine=True)
                self.layer_norm2 = LayerNormNCHW(renorm_shape[0], affine=True)
            else:
                self.layer_norm1 = LayerNorm(
                    renorm_shape, affine_shape=affine_shape
                )
                self.layer_norm2 = LayerNorm(
                    renorm_shape, affine_shape=affine_shape
                )
            self.renorm = True
        else:
            self.renorm = False

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        sc = self.shortcut(x)
        out += sc
        out = F.relu(out)
        if self.renorm:
            out = self.layer_norm1(out)
            out += sc
            out = self.layer_norm2(out)
        return out


# class AnotherBasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1, renorm_shape=None):
#         super(AnotherBasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(
#             in_planes,
#             planes,
#             kernel_size=3,
#             stride=stride,
#             padding=1,
#             bias=False,
#         )
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(
#             planes, planes, kernel_size=3, stride=1, padding=1, bias=False
#         )
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(
#                     in_planes,
#                     self.expansion * planes,
#                     kernel_size=1,
#                     stride=stride,
#                     bias=False,
#                 ),
#                 nn.BatchNorm2d(self.expansion * planes),
#             )
#         if renorm_shape is not None:
#             self.layer_norm1 = nn.LayerNorm(renorm_shape)
#             self.layer_norm2 = nn.LayerNorm(renorm_shape)
#             self.renorm = True
#         else:
#             self.renorm = False

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         sc = self.shortcut(x)
#         out += sc
#         if self.renorm:
#             out = self.layer_norm1(out)
#             out += sc
#             out = self.layer_norm2(out)
#         return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, in_planes, planes, stride=1, renorm_shape=None, affine_shape=None
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

        if renorm_shape is not None or affine_shape is not None:
            if (
                renorm_shape[1] == 0
                and renorm_shape[2] == 0
                and affine_shape[1] == 1
                and affine_shape[2] == 1
            ):
                self.layer_norm1 = LayerNormNCHW(renorm_shape[0], affine=True)
                self.layer_norm2 = LayerNormNCHW(renorm_shape[0], affine=True)
            else:
                self.layer_norm1 = LayerNorm(
                    renorm_shape, affine_shape=affine_shape
                )
                self.layer_norm2 = LayerNorm(
                    renorm_shape, affine_shape=affine_shape
                )
            self.renorm = True
        else:
            self.renorm = False

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        sc = self.shortcut(x)
        out += sc
        out = F.relu(out)
        if self.renorm:
            out = self.layer_norm1(out)
            out += sc
            out = self.layer_norm2(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self, block, num_blocks, num_classes=10, renorm=False, renorm_mode="C"
    ):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        if renorm:
            if renorm_mode == "C":  # works
                renorm_shapes = [
                    (64 * block.expansion, 0, 0),
                    (128 * block.expansion, 0, 0),
                    (256 * block.expansion, 0, 0),
                    (512 * block.expansion, 0, 0),
                ]
                affine_shapes = [
                    (64 * block.expansion, 1, 1),
                    (128 * block.expansion, 1, 1),
                    (256 * block.expansion, 1, 1),
                    (512 * block.expansion, 1, 1),
                ]
            elif renorm_mode == "CHW":  # does not work
                renorm_shapes = [
                    (64 * block.expansion, 32, 32),
                    (128 * block.expansion, 16, 16),
                    (256 * block.expansion, 8, 8),
                    (512 * block.expansion, 4, 4),
                ]
                affine_shapes = [
                    (64 * block.expansion, 32, 32),
                    (128 * block.expansion, 16, 16),
                    (256 * block.expansion, 8, 8),
                    (512 * block.expansion, 4, 4),
                ]
            elif renorm_mode == "CHW/C":  # works
                renorm_shapes = [
                    (64 * block.expansion, 32, 32),
                    (128 * block.expansion, 16, 16),
                    (256 * block.expansion, 8, 8),
                    (512 * block.expansion, 4, 4),
                ]
                affine_shapes = [
                    (64 * block.expansion, 1, 1),
                    (128 * block.expansion, 1, 1),
                    (256 * block.expansion, 1, 1),
                    (512 * block.expansion, 1, 1),
                ]
        else:
            renorm_shapes = [None, None, None, None]
            affine_shapes = [None, None, None, None]
        self.layer1 = self._make_layer(
            block,
            64,
            num_blocks[0],
            stride=1,
            renorm_shape=renorm_shapes[0],
            affine_shape=affine_shapes[0],
        )
        self.layer2 = self._make_layer(
            block,
            128,
            num_blocks[1],
            stride=2,
            renorm_shape=renorm_shapes[1],
            affine_shape=affine_shapes[1],
        )
        self.layer3 = self._make_layer(
            block,
            256,
            num_blocks[2],
            stride=2,
            renorm_shape=renorm_shapes[2],
            affine_shape=affine_shapes[2],
        )
        self.layer4 = self._make_layer(
            block,
            512,
            num_blocks[3],
            stride=2,
            renorm_shape=renorm_shapes[3],
            affine_shape=affine_shapes[3],
        )
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self,
        block,
        planes,
        num_blocks,
        stride,
        renorm_shape=None,
        affine_shape=None,
    ):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride,
                    renorm_shape=renorm_shape,
                    affine_shape=affine_shape,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_label, renorm=False):
    return ResNet(
        BasicBlock, [2, 2, 2, 2], num_classes=num_label, renorm=renorm
    )


def ResNet34(num_label, renorm=False):
    return ResNet(
        BasicBlock, [3, 4, 6, 3], num_classes=num_label, renorm=renorm
    )


# def AnotherResNet34(num_label, renorm=False):
#     return ResNet(
#         AnotherBasicBlock, [3, 4, 6, 3], num_classes=num_label, renorm=renorm
#     )


def ResNet50(num_label, renorm=False):
    return ResNet(
        Bottleneck, [3, 4, 6, 3], num_classes=num_label, renorm=renorm
    )


def ResNet101(num_label, renorm=False):
    return ResNet(
        Bottleneck, [3, 4, 23, 3], num_classes=num_label, renorm=renorm
    )


def ResNet152(num_label, renorm=False):
    return ResNet(
        Bottleneck, [3, 8, 36, 3], num_classes=num_label, renorm=renorm
    )


def build_model(args, device, state_dict=None):
    # print("==> Building model..")
    if args.model == "net":
        net = Net(args.num_blocks, args.num_label, 3, True, "pad", args.renorm)
    elif args.model == "resnet" or args.model == "resnet34":
        net = ResNet34(args.num_label, args.renorm)
    elif args.model == "resnet18":
        net = ResNet18(args.num_label, args.renorm)
    elif args.model == "resnet50":
        net = ResNet50(args.num_label, args.renorm)
    elif args.model == "resnet101":
        net = ResNet101(args.num_label, args.renorm)
    elif args.model == "resnet152":
        net = ResNet152(args.num_label, args.renorm)
    # else:
    #     assert args.model == "resnet_another", f"Unknown model {args.model}"
    #     net = AnotherResNet34(args.num_label, args.renorm)

    if state_dict is not None:
        net.load_state_dict(state_dict)
    net = net.to(device)
    # if device == "cuda":
    #     net = torch.nn.DataParallel(net)
    #     cudnn.benchmark = True

    return net


def create_optimizer(args, model_params):
    if args.optim == "sgd":
        return optim.SGD(
            model_params,
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optim == "adam":
        return optim.AdamW(
            model_params,
            args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
    else:
        raise NotImplementedError()


class MonumentLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, milestones, lrs, last_epoch=-1):
        assert len(milestones) + 1 == len(
            lrs
        ), f"Lrs should be a list one item more than milestons. Got {lrs} and {milestones}"
        self.milestones = milestones
        self.lrs = lrs
        super(MonumentLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            self.lrs[bisect_right(self.milestones, self.last_epoch)]
            for _ in self.base_lrs
        ]


def get_learning_rate(scheduler):
    if isinstance(scheduler, optim.lr_scheduler._LRScheduler):
        return get_learning_rate(scheduler.optimizer)
    else:
        assert isinstance(scheduler, optim.Optimizer)
        optimizer = scheduler
        return optimizer.param_groups[0]["lr"]


def train(net, epoch, device, data_loader, optimizer, criterion):

    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(
        tqdm.tqdm(data_loader, dynamic_ncols=True, leave=False)
    ):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total
    print("Epoch {} | train acc {:.3f}".format(epoch, accuracy))

    return accuracy


def test(net, epoch, device, data_loader, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(
            tqdm.tqdm(data_loader, dynamic_ncols=True, leave=False)
        ):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total
    print("Epoch {} | test acc {:.3f}".format(epoch, accuracy))

    return accuracy


def main():
    args = get_args()

    train_loader, test_loader = build_dataset(args)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    ckpter = Checkpointer(args.save_dir)

    ckpt = ckpter.load(args.restore)

    # def load(epoch):
    #     if epoch >= 0:
    #         ckpt_name = os.path.join(
    #             args.save_dir, "checkpoint_{}.pt".format(epoch)
    #         )
    #         if os.path.exists(ckpt_name):
    #             state_dict = torch.load(ckpt_name, "cpu")
    #             print("checkpoint {} loaded".format(ckpt_name))
    #             return (
    #                 state_dict["net"],
    #                 state_dict["optimizer"],
    #                 state_dict["last_epoch"],
    #             )
    #         print("checkpoint {} does not exist".format(ckpt_name))
    #     return None, None, None

    # ckpt = load(args.restore)
    if ckpt[0] is None:
        net = build_model(args, device)
        optimizer = create_optimizer(args, net.parameters())
        last_epoch = -1
    else:
        net = build_model(args, device, ckpt[0])
        optimizer = create_optimizer(args, net.parameters())
        optimizer.load_state_dict(ckpt[1])
        last_epoch = ckpt[2]

    criterion = nn.CrossEntropyLoss()
    if args.scheduler == "normal":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, [21, 123], gamma=args.gamma, last_epoch=last_epoch
        )
        #scheduler = optim.lr_scheduler.MultiStepLR(
        #    optimizer, [82, 123], gamma=args.gamma, last_epoch=last_epoch
        #)
    elif args.scheduler == "renorm1":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, [16, 82, 123], gamma=args.gamma, last_epoch=last_epoch
        )
    elif args.scheduler == "renorm2":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            [16, 40, 82, 123],
            gamma=args.gamma,
            last_epoch=last_epoch,
        )
    elif args.scheduler == "renorm3":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, [32, 82, 123], gamma=args.gamma, last_epoch=last_epoch
        )
    else:
        assert (
            args.scheduler == "custom"
        ), f"unknown scheduler value {args.scheduler}"
        if args.lrs:
            scheduler = MonumentLR(
                optimizer, args.scheduler_steps, args.lrs, last_epoch=last_epoch
            )
        else:
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                args.scheduler_steps,
                gamma=args.gamma,
                last_epoch=last_epoch,
            )

    # def save(last_epoch):
    #     start = time.time()
    #     ckpt_name = os.path.join(
    #         args.save_dir, "checkpoint_{}.pt".format(epoch)
    #     )
    #     state_dict = {
    #         "net": net.state_dict(),
    #         "optimizer": optimizer.state_dict(),
    #         "last_epoch": last_epoch,
    #     }
    #     torch.save(state_dict, ckpt_name)
    #     print(
    #         "checkpoint saved to {} (used {:.3f}s)".format(
    #             ckpt_name, time.time() - start
    #         )
    #     )

    writer = SummaryWriter(args.save_dir)
    # train_writer = SummaryWriter(
    #     os.path.join(args.save_dir, "train")
    # )
    # test_writer = SummaryWriter(
    #     os.path.join(args.save_dir, "test")
    # )

    writer.add_text(
        "model",
        str({"args": args, "model": net, "optim": optimizer}),
        last_epoch + 1,
    )

    writer.add_graph(
        net,
        torch.zeros([2, 3, 32, 32], dtype=torch.float32, device=device),
        verbose=False,
    )

    print("Arguments {}".format(args))
    print("Model {}".format(net))
    print("Optim {}".format(optimizer))

    for epoch in range(last_epoch + 1, 205):
        writer.add_scalar("lr", get_learning_rate(scheduler), epoch)
        train_acc = train(
            net, epoch, device, train_loader, optimizer, criterion
        )
        test_acc = test(net, epoch, device, test_loader, criterion)

        scheduler.step()

        writer.add_scalar("acc/train", train_acc, epoch)
        writer.add_scalar("acc/test", test_acc, epoch)

        state_dict = {
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "last_epoch": epoch,
        }
        ckpter.save(epoch, state_dict, test_acc)
        # save(epoch)


if __name__ == "__main__":
    main()
