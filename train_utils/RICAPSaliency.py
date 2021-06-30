import numpy as np
import torch
import cv2

class SaliencyRICAP():
    def __init__(self, criterion):
        self.beta = 1
        self.criterion = criterion

    def ricap(self, images, targets, args):

        beta = self.beta  # hyperparameter

        # size of image
        I_x, I_y = images.size()[2:]

        # generate boundary Crop Size (w, h)
        w = int(np.round(I_x//2 * np.random.beta(beta, beta)))
        h = int(np.round(I_y//2 * np.random.beta(beta, beta)))
        w_ = [w*2, I_x - w*2, w*2, I_x - w*2]
        h_ = [h*2, h*2, I_y - h*2, I_y - h*2]

        cropped_images = {}
        c_ = {}
        W_ = {}
        # select four images
        for k in range(4):
            # images rand index
            index = torch.randperm(images.size(0))
            temp_images = torch.empty(images.size(0), images.size(1), w_[k]//2*2, h_[k]//2*2)
            # compute saliency each image
            for ind in range(images.size(0)):
                # compute saliency
                temp_img = images[index[ind]].cpu().numpy().transpose(1, 2, 0)
                saliency = cv2.saliency.StaticSaliencyFineGrained_create()
                (success, saliencyMap) = saliency.computeSaliency(temp_img)
                saliencyMap = (saliencyMap * 255).astype("uint8")

                # compute maximum saliency
                maximum_indices = np.unravel_index(np.argmax(saliencyMap, axis=None), saliencyMap.shape)
                x_k = maximum_indices[0]
                y_k = maximum_indices[1]

                # x_k, y_k 위치 조정 해주기
                # 위에서 사전에 정한 w_, h_(crop할 크기)를 더했을 때 이미지의 크기를 초과하거나 0보다 작은 경우를 대비한 보정
                if x_k + (w_[k]//2) - 32 > 0 :
                    x_k -= (x_k + (w_[k]//2) - 32)
                elif x_k - (w_[k]//2) < 0:
                    x_k -= (x_k - (w_[k]//2))

                if y_k + (h_[k]//2) - 32 > 0 :
                    y_k -= (y_k + (h_[k]//2) - 32)
                elif y_k - (h_[k]//2) < 0:
                    y_k -= (y_k - (h_[k]//2))

                # x_k, y_k (Maximum Saliency 위치)로 부터 w_, h_ 만큼 crop
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

    def ricap_criterion(self, outputs, c_, W_):
        loss = sum([W_[k] * self.criterion(outputs, c_[k]) for k in range(4)])
        return loss