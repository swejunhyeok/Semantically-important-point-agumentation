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

from scipy.stats import bernoulli
from torch.autograd import Variable

import models.resnet as RN

#######################################

from utils.utils import AverageMeter, adjust_learning_rate, get_learning_rate, accuracy
from train_utils.RICAP import TrainerRICAP
from train_utils.cutmix import rand_bbox, get_lam
from train_utils.keepfo_avggridmix import KeepFOAVGGridmix
from train_utils.SaliencyRICAP import SaliencyRICAP
from train_utils.SaliencyGridMix import SaliencyGridMix
from train_utils.SaliencyMix import saliency_bbox
from train_utils.MultiImageSaliency import MultiImageSaliency

from salicenymapper.SaliencyMapper import SaliencyMapper, AVGSaliencyMapper

from dataset.VanillaBackprop import VanillaBackprop, convert_to_grayscale

import warnings

warnings.filterwarnings("ignore")

def get_train(args, progress_bar, model, criterion, optimizer, epoch, salinecy_model):
    train_loss, train_acc1 = original_train(progress_bar, model, criterion, optimizer, epoch, args, salinecy_model)
    return train_loss, train_acc1

def original_train(progress_bar, model, criterion, optimizer, epoch, args, salinecy_model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    is_train = True

    # switch to train mode
    model.train()

    if args.augmentation.startswith('SaliencyFeaturemap') or args.augmentation.startswith('FeatureCutmix'):
        if args.depth == 50:
            if args.dataset == 'cifar10':
                salinecy_model = RN.ResNet(args.dataset, args.depth, 10, args.bottleneck)
                salinecy_model.load_state_dict(
                    torch.load('../checkpoint/resnet_50_cifar10_cutmix_2021-03-24/model_best.pth.tar')['state_dict']
                )
            elif args.dataset == 'cifar100': 
                salinecy_model = RN.ResNet(args.dataset, args.depth, 100, args.bottleneck)
                salinecy_model.load_state_dict(
                    torch.load('../checkpoint/resnet_50_cifar100_cutmix_2021-03-23/model_best.pth.tar')['state_dict']
                )
            else:
                raise Exception('unknown dataset: {}'.format(args.dataset))
        elif args.depth == 18:
            if args.dataset == 'cifar10':
                salinecy_model = RN.ResNet(args.dataset, args.depth, 10, args.bottleneck)
                salinecy_model.load_state_dict(
                    torch.load('../checkpoint/resnet_18_cifar10_cutmix_2021-03-25/model_best.pth.tar')['state_dict']
                )
            elif args.dataset == 'cifar100': 
                salinecy_model = RN.ResNet(args.dataset, args.depth, 100, args.bottleneck)
                salinecy_model.load_state_dict(
                    torch.load('../checkpoint/resnet_18_cifar100_cutmix_2021-03-25/model_best.pth.tar')['state_dict']
                )
            else:
                raise Exception('unknown dataset: {}'.format(args.dataset))
        elif args.depth == 101:
            if args.dataset == 'cifar10':
                salinecy_model = RN.ResNet(args.dataset, args.depth, 10, args.bottleneck)
                salinecy_model.load_state_dict(
                    torch.load('../checkpoint/resnet_101_cifar10_cutmix_2021-03-26/model_best.pth.tar')['state_dict']
                )
            elif args.dataset == 'cifar100': 
                salinecy_model = RN.ResNet(args.dataset, args.depth, 100, args.bottleneck)
                salinecy_model.load_state_dict(
                    torch.load('../checkpoint/resnet_101_cifar100_cutmix_2021-03-26/model_best.pth.tar')['state_dict']
                )
            else:
                raise Exception('unknown dataset: {}'.format(args.dataset))
        salinecy_model = salinecy_model.cuda()
        salinecy_model.eval()
        VBP = VanillaBackprop(salinecy_model)

    end = time.time()
    current_LR = get_learning_rate(optimizer)[0]
    for i, (input, target) in enumerate(progress_bar):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.augmentation.startswith('RICAP'):
            r = np.random.rand(1)
            if r < 0.5:
                input = input.cuda()
                target = target.cuda()

                ricap = TrainerRICAP(criterion)
                input, (c_, W_) = ricap.ricap(input, target)
                input = input.cuda()
                
                output = model(input, is_train)

                loss = ricap.ricap_criterion(output, c_, W_)
            else:
                input = input.cuda()
                target = target.cuda()
                
                output = model(input, is_train)
                loss = criterion(output, target)
        elif args.augmentation.startswith('cutmix'):
            r = np.random.rand(1)
            if 1.0 > 0 and r < 0.5:
                # generate mixed sample
                input = input.cuda()
                target = target.cuda()

                lam = np.random.beta(1.0, 1.0)
                rand_index = torch.randperm(input.size()[0]).cuda()
                target_a = target
                target_b = target[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
                input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

                output = model(input, is_train)

                loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
            else:
                input = input.cuda()
                target = target.cuda()

                output = model(input, is_train)

                loss = criterion(output, target)
        elif args.augmentation.startswith('gridmix'):
            input = input.cuda()
            target = target.cuda()

            if args.augmentation == 'gridmix2x2':
                data_bern = bernoulli.rvs(size=4,p=0.8)
                rand_index = []
                rand_index = torch.randperm(input.size()[0]).cuda()
                target_a = target
                target_b = target[rand_index]
                cnt = 0
                lam = 0
                for bern in data_bern:
                    if bern == 0:
                        input[:, :, (cnt//2) * 16:(cnt//2 + 1) * 16, (cnt%2) * 16:(cnt%2 + 1) * 16] = input[rand_index, :, (cnt//2) * 16:(cnt//2 + 1) * 16, (cnt%2) * 16:(cnt%2 + 1) * 16]
                    else:
                        lam+=1
                    cnt += 1
                # adjust lambda to exactly match pixel ratio
                lam /= 4
                grid, output = model(input, is_train)
                global_loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
                local_loss = 0
                for index, bern in enumerate(data_bern):
                    if bern == 0:
                        local_loss += criterion(torch.squeeze(grid[:,:,index//2,index%2]), target_b)
                    else:
                        local_loss += criterion(torch.squeeze(grid[:,:,index//2,index%2]), target_a)
                local_loss /= 4
                loss = global_loss+local_loss
            elif args.augmentation == 'gridmix4x4multi':
                data_bern = bernoulli.rvs(size=16,p=0.8)
                rand_index = []
                target_a = target
                cnt = 0
                lam = 0
                num = 0
                for bern in data_bern:
                    if bern == 0:
                        rand_index.append(torch.randperm(input.size()[0]).cuda())
                        input[:, :, (cnt//4) * 8:(cnt//4 + 1) * 8, (cnt%4) * 8:(cnt%4 + 1) * 8] = input[rand_index[num], :, (cnt//4) * 8:(cnt//4 + 1) * 8, (cnt%4) * 8:(cnt%4 + 1) * 8]
                        num += 1
                    else:
                        lam+=1
                    cnt += 1
                # adjust lambda to exactly match pixel ratio
                lam /= 16
                grid, output = model(input, is_train)
                global_loss = criterion(output, target_a) * lam
                for index in range(len(rand_index)):
                    global_loss += criterion(output, target[rand_index[index]]) / 16
                local_loss = 0
                cnt = 0
                for index, bern in enumerate(data_bern):
                    if bern == 0:
                        local_loss += criterion(torch.squeeze(grid[:,:,index//4,index%4]), target[rand_index[cnt]])
                        cnt += 1
                    else:
                        local_loss += criterion(torch.squeeze(grid[:,:,index//4,index%4]), target_a)
                local_loss /= 16
                loss = global_loss+local_loss
            elif args.augmentation == 'gridmix2x2multi':
                data_bern = bernoulli.rvs(size=4,p=0.8)
                rand_index = []
                target_a = target
                cnt = 0
                lam = 0
                num = 0
                for bern in data_bern:
                    if bern == 0:
                        rand_index.append(torch.randperm(input.size()[0]).cuda())
                        input[:, :, (cnt//2) * 16:(cnt//2 + 1) * 16, (cnt%2) * 16:(cnt%2 + 1) * 16] = input[rand_index[num], :, (cnt//2) * 16:(cnt//2 + 1) * 16, (cnt%2) * 16:(cnt%2 + 1) * 16]
                        num += 1
                    else:
                        lam+=1
                    cnt += 1
                # adjust lambda to exactly match pixel ratio
                lam /= 4
                grid, output = model(input, is_train)
                global_loss = criterion(output, target_a) * lam
                for index in range(len(rand_index)):
                    global_loss += criterion(output, target[rand_index[index]]) / 4
                local_loss = 0
                cnt = 0
                for index, bern in enumerate(data_bern):
                    if bern == 0:
                        local_loss += criterion(torch.squeeze(grid[:,:,index//2,index%2]), target[rand_index[cnt]])
                        cnt += 1
                    else:
                        local_loss += criterion(torch.squeeze(grid[:,:,index//2,index%2]), target_a)
                local_loss /= 4
                loss = global_loss+local_loss
            elif args.augmentation.startswith('gridmix4x4'):
                data_bern = bernoulli.rvs(size=16,p=0.8)
                rand_index = []
                rand_index = torch.randperm(input.size()[0]).cuda()
                target_a = target
                target_b = target[rand_index]
                cnt = 0
                lam = 0
                for bern in data_bern:
                    if bern == 0:
                        input[:, :, (cnt//4) * 8:(cnt//4 + 1) * 8, (cnt%4) * 8:(cnt%4 + 1) * 8] = input[rand_index, :, (cnt//4) * 8:(cnt//4 + 1) * 8, (cnt%4) * 8:(cnt%4 + 1) * 8]
                    else:
                        lam+=1
                    cnt += 1
                # adjust lambda to exactly match pixel ratio
                lam /= 16
                grid, output = model(input, is_train)
                global_loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
                local_loss = 0
                cnt = 0
                for index, bern in enumerate(data_bern):
                    if bern == 0:
                        local_loss += criterion(torch.squeeze(grid[:,:,index//4,index%4]), target_b)
                        cnt += 1
                    else:
                        local_loss += criterion(torch.squeeze(grid[:,:,index//4,index%4]), target_a)
                local_loss /= 16
                loss = global_loss+local_loss
        elif args.augmentation.startswith('keepfo_avg_gridmix'):
            input = input.cuda()
            target = target.cuda()
            if args.augmentation.endswith('v2'):
                r = np.random.rand(1)
                if r < 0.5:
                    bbox = AVGSaliencyMapper(salinecy_model, input, target)
                    bbox = torch.FloatTensor(bbox).cuda()
                    keepfo = KeepFOAVGGridmix(criterion, args)

                    input, target_b, grid_num, shuffle_grid = keepfo.keepfo(input, target, bbox)
                    input = input.cuda()
                    grid, output = model(input, is_train)
                    loss = keepfo.keepfo_criterion(output, target, target_b, grid_num, shuffle_grid, grid)
                else:
                    _, output = model(input, is_train)
                    loss = criterion(output, target)
            else:
                bbox = AVGSaliencyMapper(salinecy_model, input, target)
                bbox = torch.FloatTensor(bbox).cuda()
                keepfo = KeepFOAVGGridmix(criterion, args)

                input, target_b, grid_num, shuffle_grid = keepfo.keepfo(input, target, bbox)
                input = input.cuda()
                grid, output = model(input, is_train)
                loss = keepfo.keepfo_criterion(output, target, target_b, grid_num, shuffle_grid, grid)
        elif args.augmentation.startswith('SalinecyRICAP'):
            input = input.cuda()
            target = target.cuda()
            
            if args.augmentation.endswith('v2'):
                r = np.random.rand(1)
                if r < 0.5:
                    ricap = SaliencyRICAP(criterion)
                    input, (c_, W_) = ricap.ricap(input, target, args)
                    input = input.cuda()
                    
                    output = model(input, is_train)

                    loss = ricap.ricap_criterion(output, c_, W_)
                else:
                    output = model(input, is_train)
                    loss = criterion(output, target)
            else:
                r = np.random.rand(1)
                if r < 0.5:
                    ricap = SaliencyRICAP(criterion)
                    input, (c_, W_) = ricap.ricap(input, target, args)
                    input = input.cuda()
                    
                    output = model(input, is_train)

                    loss = ricap.ricap_criterion(output, c_, W_)
                else:
                    output = model(input, is_train)
                    loss = criterion(output, target)
        elif args.augmentation.startswith('SaliencyGridMix'):
            input = input.cuda()
            target = target.cuda()

            if args.augmentation.endswith('v2'):
                r = np.random.rand(1)
                if r < 0.5:
                    keepfo = SaliencyGridMix(criterion, args)
                    images, target_b, grid_num, shuffle_grid = keepfo.keepfo(input, target)
                    input = input.cuda()
                    
                    grid, output = model(input, is_train)

                    loss = keepfo.keepfo_criterion_v2(output, target, target_b, grid_num, shuffle_grid, grid)
                else:
                    _, output = model(input, is_train)
                    loss = criterion(output, target)
            else:
                r = np.random.rand(1)
                if r < 0.5:
                    keepfo = SaliencyGridMix(criterion, args)
                    images, target_b, grid_num, shuffle_grid = keepfo.keepfo(input, target)
                    input = input.cuda()
                    
                    grid, output = model(input, is_train)

                    loss = keepfo.keepfo_criterion(output, target, target_b, grid_num, shuffle_grid, grid)
                else:
                    _, output = model(input, is_train)
                    loss = criterion(output, target)
        elif args.augmentation.startswith('SaliencyFeaturemap'):
            r = np.random.rand(1)
            if r < 0.5:
                input = input.cuda()
                target = target.cuda()
                vanilla_grads = VBP.generate_gradients(input, target)

                # Convert to grayscale
                grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads)
                index = torch.randperm(input.size(0))
                target_b = target[index]
                beta = 1
                w = int(np.round(input.size(2)* np.random.beta(beta, beta)))

                output = model(input, is_train, True, index, w, w, 1, grayscale_vanilla_grads, args)
                lam = 1 - ((w//2) * (w//2) / (input.size()[-1] * input.size()[-2]))
                loss = criterion(output, target) * lam + criterion(output, target_b) * (1. - lam)
            else:
                input = input.cuda()
                target = target.cuda()

                output = model(input, is_train)

                loss = criterion(output, target)
        elif args.augmentation.startswith('FeatureCutmix'):
            r = np.random.rand(1)
            if 1.0 > 0 and r < 0.5:
                # generate mixed sample
                input = input.cuda()
                target = target.cuda()

                lam = np.random.beta(1.0, 1.0)
                rand_index = torch.randperm(input.size()[0]).cuda()
                target_a = target
                target_b = target[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
                input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]

                vanilla_grads = VBP.generate_gradients(input, target)

                # Convert to grayscale
                grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads)

                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

                output = model(input, is_train, True, rand_index, (bbx2 - bbx1), (bby2 - bby1), 1, grayscale_vanilla_grads, args)

                loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
            else:
                input = input.cuda()
                target = target.cuda()

                output = model(input, is_train)

                loss = criterion(output, target)
        elif args.augmentation.startswith('SaliencyMix'):
            r = np.random.rand(1)
            if r < 0.5:
                input = input.cuda()
                target = target.cuda()

                lam = np.random.beta(1.0, 1.0)
                rand_index = torch.randperm(input.size()[0]).cuda()
                target_a = target
                target_b = target[rand_index]
                bbx1, bby1, bbx2, bby2 = saliency_bbox(input[rand_index[0]], lam)
                input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
                output = model(input, is_train)

                loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
            else:
                input = input.cuda()
                target = target.cuda()

                output = model(input, is_train)

                loss = criterion(output, target)
        elif args.augmentation.startswith('MultiImageSaliency'):
            input = input.cuda()
            target = target.cuda()
            r = np.random.rand(1)
            if r < 0.5:
                keepfo = MultiImageSaliency(criterion)
                input, (c_, W_) = keepfo.ricap(input, target, args)
                input = input.cuda()
                
                grid, output = model(input, is_train)

                loss = keepfo.ricap_criterion(output, c_, W_, grid)
            else:
                _, output = model(input, is_train)
                loss = criterion(output, target)
        else:
            input = input.cuda()
            target = target.cuda()

            output = model(input, is_train)

            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        # compute gradient and do SGD step
        model.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        progress_bar.set_description('Epoch: [{0}/{1}][{2}/{3}] '
                  'Loss {loss.val:.2f} ({loss.avg:.2f}) '
                  'Top 1-acc {top1.val:.2f} ({top1.avg:.2f}) '
                  'Top 5-acc {top5.val:.2f} ({top5.avg:.2f})'.format(
                epoch, args.epochs, i, len(progress_bar), loss=losses, top1=top1, top5=top5))

    progress_bar.write('* Train Epoch: [{0}/{1}]\t Top 1-acc {top1.avg:.3f}  Top 5-acc {top5.avg:.3f}\t Train Loss {loss.avg:.3f}\t LR: {LR:.6f}\t Time {batch_time.avg:.3f} \n'.format(
        epoch, args.epochs, top1=top1, top5=top5, loss=losses, LR=current_LR, batch_time=batch_time))

    return losses.avg, top1.val
