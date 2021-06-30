import numpy as np
import torch
import cv2

class MultiImageSaliency():
    def __init__(self, criterion):
        self.beta = 1
        self.criterion = criterion

    def ricap(self, images, targets, args):

        beta = self.beta  # hyperparameter

        # size of image
        I_x, I_y = images.size()[2:]

        # generate boundary position (w, h)
        w_ = [I_x//2, I_x//2, I_x//2, I_x//2]
        h_ = [I_y//2, I_y//2, I_y//2, I_y//2]

        # select four images
        cropped_images = {}
        c_ = {}
        W_ = {}
        for k in range(4):
            index = torch.randperm(images.size(0))
            temp_images = torch.empty(images.size(0), images.size(1), w_[k]//2*2, h_[k]//2*2)
            for ind in range(images.size(0)):
                temp_img = images[index[ind]].cpu().numpy().transpose(1, 2, 0)
                saliency = cv2.saliency.StaticSaliencyFineGrained_create()
                (success, saliencyMap) = saliency.computeSaliency(temp_img)
                saliencyMap = (saliencyMap * 255).astype("uint8")

                maximum_indices = np.unravel_index(np.argmax(saliencyMap, axis=None), saliencyMap.shape)
                x_k = maximum_indices[0]
                y_k = maximum_indices[1]

                if x_k + (w_[k]//2) - I_x > 0 :
                    x_k -= (x_k + (w_[k]//2) - I_x)
                elif x_k - (w_[k]//2) < 0:
                    x_k -= (x_k - (w_[k]//2))

                if y_k + (h_[k]//2) - I_y > 0 :
                    y_k -= (y_k + (h_[k]//2) - I_y)
                elif y_k - (h_[k]//2) < 0:
                    y_k -= (y_k - (h_[k]//2))
                temp_images[ind] = images[index[ind]][:, x_k-(w_[k]//2):x_k + (w_[k]//2), y_k - (h_[k]//2):y_k + (h_[k]//2)]
            cropped_images[k] = temp_images
            c_[k] = targets[index]
            W_[k] = (w_[k] * h_[k]) / (I_x * I_y)

        # patch cropped images
        patched_images = torch.cat(
            (torch.cat((cropped_images[0], cropped_images[1]), 2),
             torch.cat((cropped_images[2], cropped_images[3]), 2)),
            3)

        targets = (c_, W_)
        return patched_images, targets

    def ricap_criterion(self, outputs, c_, W_, grid):
        loss = sum([W_[k] * self.criterion(outputs, c_[k]) for k in range(4)])
        local_loss=0
        for i, c in enumerate(c_):
            local_loss += self.criterion(grid[:,:,i//2,i%2], c_[i])
        local_loss /= 4
        loss += local_loss
        return loss