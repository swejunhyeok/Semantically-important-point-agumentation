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

#######################################

from utils.utils import AverageMeter, adjust_learning_rate, get_learning_rate, accuracy

import warnings

warnings.filterwarnings("ignore")

def get_validate(args, val_loader, model, criterion, epoch):
    acc1, acc5, val_loss = validate(val_loader, model, criterion, epoch, args)
    return acc1, acc5, val_loss

def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    is_train = False

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        with torch.no_grad():
            if args.augmentation.startswith('gridmix') or args.augmentation.startswith('SaliencyGridMix') or args.augmentation.startswith('MultiImageSaliency'):
                _, output = model(input_var, is_train)
            else:
                output = model(input_var, is_train)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))

        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('* Test Epoch: [{0}/{1}]\t Top 1-acc {top1.avg:.3f}  Top 5-acc {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=top1, top5=top5, loss=losses))
    return top1.avg, top5.avg, losses.avg