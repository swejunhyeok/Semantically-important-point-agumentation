import os
import time
from tqdm import tqdm
import numpy as np
import random

import cv2

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import models.resnet as RN
import models.resnext as RXN
import models.svhn as SVHN
import models.wideresnet as WRN

#######################################
from dataset.dataloader import get_dataloader
from train_utils.train import get_train
from train_utils.validate import get_validate
from salicenymapper.SaliencyMapper import SalinecyMapperModel

from utils.utils import rand_bbox, save_checkpoint, AverageMeter, adjust_learning_rate, get_learning_rate, accuracy, CSVLogger, CSVLogger2

from args import args

import warnings

warnings.filterwarnings("ignore")

best_acc1 = 0
best_acc5 = 0

def main():
    global args, best_acc1, best_acc5, start_epoch, test_id

    if args.net_type == 'wideresnet':
        args.depth = 28

    test_id = args.net_type + '_' + str(args.depth) + '_' + args.dataset + '_' + args.augmentation + '_'  +str(args.batch_size) + '_' + str(time.strftime('%Y-%m-%d', time.localtime(time.time())))
    print(test_id)

    csv_path = '../logs/'
    checkpoint_path = "../checkpoint/"
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    train_loader, val_loader, numberofclass = get_dataloader(args)

    print("=> creating model '{}'".format(args.net_type))
    if args.net_type == 'resnet':
        if args.augmentation.startswith('MultiImageSaliency'):
            print('GridMixResnet')
            model = RN.GridMixResNet(args.dataset, args.depth, numberofclass, args.bottleneck, args)
        else:
            model = RN.ResNet(args.dataset, args.depth, numberofclass, args.bottleneck)  # for ResNet
    elif args.net_type == 'resnext':
        model = RXN.ResNet(args.dataset, args.depth, numberofclass)
    elif args.net_type == 'wideresnet':
        model = WRN.WideResNet(depth=28, num_classes=numberofclass, widen_factor=10, dropRate=0.3)
    elif args.net_type == 'svhn':
        model = SVHN.stl10(args.depth)
    else:
        raise Exception('unknown network architecture: {}'.format(args.net_type))

    # if args.augmentation.startswith('SaliencyFeaturemap'):
    #     salinecy_model = RN.ResNet(args.dataset, args.depth, numberofclass, args.bottleneck)
    #     if args.dataset == 'cifar10':
    #        salinecy_model.load_state_dict(
    #             torch.load('../checkpoint/resnet_50_cifar10_cutmix_2021-03-24/model_best.pth.tar')['state_dict']
    #         )
    #     elif args.dataset == 'cifar100': 
    #         salinecy_model.load_state_dict(
    #             torch.load('../checkpoint/resnet_50_cifar100_cutmix_2021-03-23/model_best.pth.tar')['state_dict']
    #         )
    #     else:
    #         raise Exception('unknown dataset: {}'.format(args.dataset))
    #     salinecy_model = salinecy_model.cuda()
    # else:
    #     salinecy_model = None
    salinecy_model = None

    model = model.cuda()

    if args.net_type == 'svhn':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=True)

    if args.resume:
        checkpoint = torch.load('../checkpoint/' + test_id + '/model_best.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        filename = csv_path + '/' + test_id + '.csv'
        csv_logger = CSVLogger2(args=args, fieldnames=['epoch', 'train_loss', 'train_acc1', 'test_loss', 'test_acc1', 'test_acc5'], filename=filename)
    else:
        start_epoch = 0
        filename = csv_path + '/' + test_id + '.csv'
        csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_loss', 'train_acc1', 'test_loss', 'test_acc1', 'test_acc5'], filename=filename)

    print('the number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    cudnn.benchmark = True

    for epoch in range(start_epoch, start_epoch + args.epochs):
        progress_bar = tqdm(train_loader)

        adjust_learning_rate(args, optimizer, epoch)

        # train for one epoch
        train_loss, train_acc1 = get_train(args, progress_bar, model, criterion, optimizer, epoch, salinecy_model)

        # evaluate on validation set
        acc1, acc5, val_loss = get_validate(args, val_loader, model, criterion, epoch)
        
        train_loss = '%.4f' % (train_loss)
        train_acc1 = '%.4f' % (train_acc1)

        val_loss = '%.4f' % (val_loss)
        test_acc1 = '%.4f' % (acc1)
        test_acc5 = '%.4f' % (acc5)

        # remember best prec@1 and save checkpoint
        is_best = acc1 >= best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            best_acc1 = acc1
            best_acc5 = acc5
            save_checkpoint({
                'epoch': epoch,
                'arch': args.net_type,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'best_acc5': best_acc5,
                'optimizer': optimizer.state_dict(),
            }, test_id)
            tqdm.write('save check point')

        tqdm.write('Current best accuracy (top-1 and 5 Accuracy): {:.2f} / {:.2f}'.format(best_acc1, best_acc5))

        row = {'epoch': str(epoch), 'train_loss': str(train_loss), 'train_acc1': str(train_acc1), 'test_loss': str(val_loss), 'test_acc1': str(acc1), 'test_acc5': str(acc5)}
        csv_logger.writerow(row)

    print('Best accuracy (top-1 and 5 Accuracy):', best_acc1, best_acc5)
    csv_logger.close()

if __name__ == '__main__':
    main()