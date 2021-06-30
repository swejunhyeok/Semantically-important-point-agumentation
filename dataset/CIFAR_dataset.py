from PIL import Image
import os
from tqdm import tqdm
import os.path
import numpy as np
import pickle
import sys
from typing import Any, Callable, Optional, Tuple

from torch.utils.data import Dataset
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import VisionDataset
import torchvision.models as models

from dataset.VanillaBackprop import VanillaBackprop, convert_to_grayscale
import models.resnet as RN
from train_utils.cutout import Cutout

class CIFAR10(VisionDataset):
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, args= None):
        self.train = train  # training set or test set
        self.args = args
        self.root = root
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []
        self.transform=transform

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        target = self.targets[index]
        image = self.data[index]
        image = image.astype(np.uint8)
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        
        image = torch.FloatTensor(image)

        return image, target

    def __len__(self):
        return len(self.data)

class CIFAR100(CIFAR10):
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

def get_CIFAR_dataloader(args):
    n_holes = 1
    if args.dataset == 'cifar100': 
        length = 8
    elif args.dataset == 'cifar10': 
        length = 16
    args.lr = 0.25
    args.momentum = 0.9
    args.weight_decay = 1e-4
    args.batch_size = 64
    args.epochs = 300
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    if args.augmentation == 'baseline':
        print("Augmentation Apply Base")
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    elif args.augmentation == 'cutout':
        print("Augmentation Apply cutout")
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            Cutout(n_holes=n_holes, length=length)
        ])
    else:
        print("Augmentation Apply augmentation")
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if args.dataset == 'cifar100':
        train_loader = torch.utils.data.DataLoader(
            CIFAR100('../data', train=True, download=True, transform=transform_train, args=args),
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)
        val_loader = torch.utils.data.DataLoader(
            CIFAR100('../data', train=False, download=True, transform=transform_test),
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)
        numberofclass = 100
    elif args.dataset == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            CIFAR10('../data', train=True, download=True, transform=transform_train, args=args),
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)
        val_loader = torch.utils.data.DataLoader(
            CIFAR10('../data', train=False, download=True, transform=transform_test),
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)
        numberofclass = 10
    else:
        raise Exception('unknown dataset: {}'.format(args.dataset))

    return train_loader, val_loader, numberofclass