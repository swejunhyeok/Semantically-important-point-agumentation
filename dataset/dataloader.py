from dataset.CIFAR_dataset import get_CIFAR_dataloader
from dataset.ImageNet_dataset import get_ImageNet_dataloader
from dataset.STL_dataset import get_STL10_dataloader

import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
from utils.utils import ColorJitter, Lighting

def get_dataloader(args):
    if args.dataset.startswith('cifar'):
        train_loader, val_loader, numofclasses = get_CIFAR_dataloader(args)
    elif args.dataset == 'imagenet':
        # Image path
        traindir = os.path.join('../../dataset/tiny_imagenet/train')
        valdir = os.path.join('../../dataset/tiny_imagenet/val/images')

        # Preprocessing
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        jittering = ColorJitter(brightness=0.4, contrast=0.4,
                                      saturation=0.4)
        lighting = Lighting(alphastd=0.1,
                                  eigval=[0.2175, 0.0188, 0.0045],
                                  eigvec=[[-0.5675, 0.7192, 0.4009],
                                          [-0.5808, -0.0045, -0.8140],
                                          [-0.5836, -0.6948, 0.4203]])
        if args.augmentation == 'baseline':
            # Dataloader 
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.Resize(64),
                    transforms.ToTensor(),
                    normalize,
                ]))
        else:
            # Dataloader 
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(64),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    jittering,
                    lighting,
                    normalize,
                ]))

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        
        numofclasses = 200
    elif args.dataset == 'stl10':
        train_loader, val_loader, numofclasses = get_STL10_dataloader(args)
    
    return train_loader, val_loader, numofclasses