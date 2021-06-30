import numpy as np
import torch
import random
from scipy.stats import bernoulli
import cv2

class SaliencyGridMix():
    def __init__(self, criterion, args):
        self.args= args
        self.criterion = criterion
        if self.args.augmentation.startswith('SaliencyGridMix_2x2'):
            self.cell_num = 2
        elif self.args.augmentation.startswith('SaliencyGridMix_4x4'):
            self.cell_num = 4

    def keepfo(self, images, targets):
        # size of image
        I_x, I_y = images.size()[2:]

        rand_index = torch.randperm(images.size()[0]).cuda()

        if self.args.augmentation.endswith('v1'):
            temp_img = images[rand_index[0]].cpu().numpy().transpose(1, 2, 0)
            saliency = cv2.saliency.StaticSaliencyFineGrained_create()
            (success, saliencyMap) = saliency.computeSaliency(temp_img)
            saliencyMap = (saliencyMap * 255).astype("uint8")
            arr_2x2 = np.zeros((self.cell_num*self.cell_num))
            for x in range(saliencyMap.shape[0]):
                for y in range(saliencyMap.shape[1]):
                    arr_2x2[int((x//(I_x/self.cell_num) * self.cell_num) + y//(I_y/self.cell_num))] += saliencyMap[x][y]
            arg_arr_2x2 = np.argpartition(arr_2x2, (1, self.cell_num - 1))
            arg_arr_2x2 = np.asarray(arg_arr_2x2)

            if self.args.augmentation.endswith('v3'):
                data_bern = bernoulli.rvs(size=self.cell_num*self.cell_num,p=0.5)
            elif self.args.augmentation.endswith('v4'):
                data_bern = bernoulli.rvs(size=self.cell_num*self.cell_num,p=0.75)
            else:
                data_bern = bernoulli.rvs(size=self.cell_num*self.cell_num,p=0.8)
            grid_num = self.cell_num*self.cell_num - np.sum(data_bern)
            shuffle_grid = []
            for i in range(self.cell_num*self.cell_num):
                shuffle_grid.append(1)
            if grid_num != 0 :
                for i in range(grid_num):
                    step_size = I_x//self.cell_num
                    best_index = int(arg_arr_2x2[self.cell_num*self.cell_num - 1 -i])
                    shuffle_grid[best_index] = 0
                    x_first_index = (best_index // self.cell_num) * step_size
                    x_second_index = (best_index // self.cell_num + 1) * step_size
                    y_first_index = (best_index % self.cell_num) * step_size
                    y_second_index = (best_index % self.cell_num + 1) * step_size
                    images[:, : , x_first_index : x_second_index, y_first_index : y_second_index] = images[rand_index, : , x_first_index : x_second_index, y_first_index : y_second_index]
        elif self.args.augmentation.endswith('v2'):
            data_bern = bernoulli.rvs(size=self.cell_num*self.cell_num,p=0.8)
            grid_num = self.cell_num*self.cell_num - np.sum(data_bern)
            shuffle_grid = []
            for ind in range(images.size()[0]):
                temp_img = images[rand_index[0]].cpu().numpy().transpose(1, 2, 0)
                saliency = cv2.saliency.StaticSaliencyFineGrained_create()
                (success, saliencyMap) = saliency.computeSaliency(temp_img)
                saliencyMap = (saliencyMap * 255).astype("uint8")
                arr_2x2 = np.zeros((self.cell_num*self.cell_num))
                for x in range(saliencyMap.shape[0]):
                    for y in range(saliencyMap.shape[1]):
                        arr_2x2[int((x//(I_x/self.cell_num) * self.cell_num) + y//(I_y/self.cell_num))] += saliencyMap[x][y]
                arg_arr_2x2 = np.argpartition(arr_2x2, (1, self.cell_num - 1))
                arg_arr_2x2 = np.asarray(arg_arr_2x2)

                temp_shuffle_grid = []
                for i in range(self.cell_num*self.cell_num):
                    temp_shuffle_grid.append(1)
                if grid_num != 0 :
                    for i in range(grid_num):
                        step_size = I_x//self.cell_num
                        best_index = int(arg_arr_2x2[self.cell_num*self.cell_num - 1 -i])
                        temp_shuffle_grid[best_index] = 0
                        x_first_index = (best_index // self.cell_num) * step_size
                        x_second_index = (best_index // self.cell_num + 1) * step_size
                        y_first_index = (best_index % self.cell_num) * step_size
                        y_second_index = (best_index % self.cell_num + 1) * step_size
                        images[ind, : , x_first_index : x_second_index, y_first_index : y_second_index] = images[rand_index[ind], : , x_first_index : x_second_index, y_first_index : y_second_index]
                shuffle_grid.append(temp_shuffle_grid)
        target_b = targets[rand_index]
        return images, target_b, grid_num, shuffle_grid

    def keepfo_criterion(self, outputs, target_a, target_b, grid_num, data_bern, grid):
        global_loss = ((self.cell_num-grid_num)/(self.cell_num*self.cell_num)) * self.criterion(outputs, target_a) + ((grid_num)/(self.cell_num*self.cell_num)) * self.criterion(outputs, target_b)
        local_loss = 0
        for i, bern in enumerate(data_bern):
            if bern == 0:
                local_loss += self.criterion(grid[:,:,i//self.cell_num,i%self.cell_num], target_b)
            else:
                local_loss += self.criterion(grid[:,:,i//self.cell_num,i%self.cell_num], target_a)
        local_loss /= self.cell_num*self.cell_num
        return global_loss+local_loss
    
    def keepfo_criterion_v2(self, outputs, target_a, target_b, grid_num, data_bern, grid):
        global_loss = ((self.cell_num-grid_num)/(self.cell_num*self.cell_num)) * self.criterion(outputs, target_a) + ((grid_num)/(self.cell_num*self.cell_num)) * self.criterion(outputs, target_b)
        local_loss = 0
        for i, bern in enumerate(data_bern):
            for j, b in enumerate(bern):
                if b == 0:
                    local_loss += self.criterion(grid[i,:,j//self.cell_num,j%self.cell_num].unsqueeze(0), target_b[i].unsqueeze(0))
                else:
                    local_loss += self.criterion(grid[i,:,j//self.cell_num,j%self.cell_num].unsqueeze(0), target_a[i].unsqueeze(0))
            local_loss /= self.cell_num*self.cell_num
        local_loss /= len(data_bern)
        return global_loss+local_loss