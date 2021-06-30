import numpy as np
import torch
import random
from scipy.stats import bernoulli

class KeepFOAVGGridmix():
    def __init__(self, criterion, args):
        self.args= args
        self.criterion = criterion
        if self.args.augmentation.startswith('keepfo_avg_gridmix_2x2'):
            self.cell_num = 2
        elif self.args.augmentation.startswith('keepfo_avg_gridmix_4x4'):
            self.cell_num = 4

    def keepfo(self, images, targets, grid):
        # size of image
        I_x, I_y = images.size()[2:]

        rand_index = torch.randperm(images.size()[0]).cuda()
        data_bern = bernoulli.rvs(size=self.cell_num*self.cell_num,p=0.8)
        grid_num = self.cell_num*self.cell_num - np.sum(data_bern)
        shuffle_grid = []
        for i in range(self.cell_num*self.cell_num):
            shuffle_grid.append(1)
        if grid_num != 0 :
            for i in range(grid_num):
                step_size = I_x//self.cell_num
                best_index = int(grid[self.cell_num*self.cell_num - 1 -i])
                shuffle_grid[best_index] = 0
                x_first_index = (best_index // self.cell_num) * step_size
                x_second_index = (best_index // self.cell_num + 1) * step_size
                y_first_index = (best_index % self.cell_num) * step_size
                y_second_index = (best_index % self.cell_num + 1) * step_size
                images[:, : , x_first_index : x_second_index, y_first_index : y_second_index] = images[rand_index, : , x_first_index : x_second_index, y_first_index : y_second_index]
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