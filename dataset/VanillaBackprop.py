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
import torchvision.models as model

from torchvision import models

def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale
    Args:
        im_as_arr (numpy arr): RGB image with shape (C,D,W,H)
    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_m = np.zeros((im_as_arr.shape[0], 1, im_as_arr.shape[2], im_as_arr.shape[3]))
    for i in range(im_as_arr.shape[0]):
        grayscale_im = np.sum(np.abs(im_as_arr[i]), axis=0)
        im_max = np.percentile(grayscale_im, 99)
        im_min = np.min(grayscale_im)
        grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
        grayscale_im = np.expand_dims(grayscale_im, axis=0)
        grayscale_im = grayscale_im - grayscale_im.min()
        grayscale_im /= grayscale_im.max()
        grayscale_m[i] = grayscale_im
    return grayscale_m

# def convert_to_grayscale(im_as_arr):
#     """
#         Converts 3d image to grayscale
#     Args:
#         im_as_arr (numpy arr): RGB image with shape (C,D,W,H)
#     returns:
#         grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
#     """
#     grayscale_m = np.zeros((im_as_arr.shape[0], 1, im_as_arr.shape[2], im_as_arr.shape[3]))
#     for i in range(im_as_arr.shape[0]):
#         grayscale_im = np.sum(np.abs(im_as_arr[i]), axis=0)
#         im_max = np.percentile(grayscale_im, 99)
#         im_min = np.min(grayscale_im)
#         grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
#         grayscale_im = np.expand_dims(grayscale_im, axis=0)
#         grayscale_im = grayscale_im - grayscale_im.min()
#         grayscale_im /= grayscale_im.max()
#         grayscale_m[i] = grayscale_im

#     grayscale_im = np.sum(np.abs(grayscale_m), axis=0)[0]
#     im_max = np.percentile(grayscale_im, 99)
#     im_min = np.min(grayscale_im)
#     grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
#     grayscale_im = np.expand_dims(grayscale_im, axis=0)
#     grayscale_im = grayscale_im - grayscale_im.min()
#     grayscale_im /= grayscale_im.max()
#     return grayscale_im

class VanillaBackprop():
    """
        Produces gradients generated with vanilla back propagation from the image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0].cpu()

        # Register hook to the first layer
        first_layer = self.model._modules.get('features')[0]
        first_layer.register_backward_hook(hook_function)

    def generate_gradients(self, input_image, target_class):
        # Forward
        model_output = self.model(input_image)
        model_output_copy = model_output.cpu()

        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(len(target_class), model_output_copy.size()[-1]).zero_()
        for i in range(len(target_class)):
            one_hot_output[i][target_class[i]] = 1
        # Backward pass
        model_output_copy.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()
        return gradients_as_arr