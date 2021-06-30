import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import cv2

import os, numpy as np
import random


import time

class ImageNetDataset(Dataset):
    def __init__(self, image_path, transform=None, is_train=False):
        self.image_path = image_path
        self.label = []
        self.image_file = []
        self.loader = default_loader
        for f in os.listdir(image_path):
            self.label.append(f)
            for i in os.listdir(image_path+'/'+f):
                self.image_file.append(image_path+'/'+f+'/'+i)

        self.num_data = len(self.image_file)
        self.transform = transform
        self.is_train = is_train
        print('Dataset Init Complete')
    
    def __getitem__(self, index):
        target = self.label.index(self.image_file[index].split('/')[3])
        image = self.loader(self.image_file[index], backend='opencv', colorSpace='BGR')
        if self.transform is not None:
            image = self.transform(image)
        
        image = torch.FloatTensor(image)

        return image, target
    
    def __len__(self):
        return int(self.num_data)


def get_ImageNet_dataloader(args, train_image_path = '../imagenet/train/', val_image_path = '../imagenet/val/', is_Hybird = False, is_mixup=False):
    input_size1 = 256
    input_size2 = 224

    transform = transforms.Compose([
            transforms.Resize(input_size1),
            transforms.CenterCrop(input_size2)
    ])
  

    image_datasets = {
        'train': DctDataset(train_image_path, transform=transform, is_train=True),
        'val' : DctDataset(val_image_path, transform=transform, is_train=False)
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=args.batch_size, 
                                             shuffle=True, num_workers=args.workers, pin_memory=True),
        'val' : torch.utils.data.DataLoader(image_datasets['val'], batch_size=args.batch_size, 
                                             shuffle=False, num_workers=args.workers, pin_memory=True)
    }

    # dataset_sizes = {
    #     'train': len(image_datasets['train']),
    #     'val': len(image_datasets['val'])
    # }
    
    return dataloaders['train'], dataloaders['val'], 1000

        