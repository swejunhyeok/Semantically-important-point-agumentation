import argparse

import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Keep 4in1 image PyTorch CIFAR-10, CIFAR-100 and ImageNet-1k Training')
parser.add_argument('--net_type', default='resnet', type=str,
                    help='networktype: resnet, resnext, svhn, vgg, wideresnet')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--dataset', dest='dataset', default='cifar10', type=str,
                    help='dataset (options: cifar10, cifar100, stl10 and imagenet)')

# Optim
parser.add_argument('--lr', '--learning-rate', default=0.25, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--augmentation', dest='augmentation', default='SalinecyRICAP_v2', type=str,
                    help='augmentation (options: baseline, augmentation, keepfo, cutmix, cutmix_region, SaliencyMix)')

# cutmix : 94.68% / 75.96%
# RICAP : 94.17%
# GridMix : 94.62%
# MultiGridMix : 94.32%
# SalinecyRICAP_v1 : 94.17%
# SalinecyRICAP_v2 : 94.50%
# SaliencyGridMix_2x2_v2 : 94.26%
# SaliencyFeaturemap_v1
# FeatureCutmix_v1
# MultiImageSaliency

# ResNet
parser.add_argument('--depth', default=50, type=int,
                    help='depth of the network (default: 32)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='to use basicblock for CIFAR datasets (default: bottleneck)')
parser.add_argument('--depth-alpha', default=300, type=float,
                    help='number of new channel increases per depth (default: 300)')      

# Keep 4 in 1 image
parser.add_argument('--keepfoThreshold', default=0.8, type=float,
                    help='hyperparameter Keep 4 in 1 Threshold')
parser.add_argument('--keepfobeta', default=1.0, type=float,
                    help='hyperparameter Keep 4 in 1 image beta')
# Train Etc.
parser.add_argument('--resume', type=bool, default=False,
                    help='restart')

args = parser.parse_args()