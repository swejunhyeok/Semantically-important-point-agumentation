import numpy as np
import torch

class KeepFO():
    def __init__(self, criterion, args):
        self.args = args
        self.beta = args.keepfobeta
        self.criterion = criterion

    def keepfo(self, images, targets, bbox):

        beta = self.beta  # hyperparameter

        # size of image
        I_x, I_y = images.size()[2:]

        # generate boundary position (w, h)
        w = int(np.round(I_x * np.random.beta(beta, beta)))
        h = int(np.round(I_y * np.random.beta(beta, beta)))
        w_ = [w, I_x - w, w, I_x - w]
        h_ = [h, h, I_y - h, I_y - h]

        # select four images
        cropped_images = {}
        c_ = {}
        W_ = {}
        for k in range(4):
            index = torch.randperm(images.size(0))
            print(bbox[index,0])
            start = min(0, I_x - (bbox[index][0] + w_[k]) + 1)
            x_k = bbox[0] - start
            print(I_y, h_[k])
            print(bbox[index][2])
            start = min(0, I_y - (bbox[index][2] + h_[k]) + 1)
            y_k = bbox[2] - start
            cropped_images[k] = images[index][:, :, x_k:x_k + w_[k], y_k:y_k + h_[k]]
            c_[k] = targets[index]
            W_[k] = (w_[k] * h_[k]) / (I_x * I_y)

        # patch cropped images
        patched_images = torch.cat(
            (torch.cat((cropped_images[0], cropped_images[1]), 2),
             torch.cat((cropped_images[2], cropped_images[3]), 2)),
            3)

        targets = (c_, W_)
        return patched_images, targets

    def keepfo_criterion(self, outputs, c_, W_):
        loss = sum([W_[k] * self.criterion(outputs, c_[k]) for k in range(4)])
        return loss