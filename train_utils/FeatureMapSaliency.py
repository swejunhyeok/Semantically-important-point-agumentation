import numpy as np
import torch
import cv2

def FeatureMapSaliency(x, rand_index, w, h, grayscale_vanilla_grads, args):
    # X 64, 64, 32, 32
    original_x = x
    for i in range(x.size(0)):
        if args.augmentation.endswith('v1'):
            Saliencyfmap = grayscale_vanilla_grads[rand_index[i],0,:,:]

            maximum_indices = np.unravel_index(np.argmax(Saliencyfmap, axis=None), Saliencyfmap.shape)
            x_k = maximum_indices[0]
            y_k = maximum_indices[1]
            
            if x_k + (w//2) - 32 > 0 :
                x_k -= (x_k + (w//2) - 32)
            elif x_k - (w//2) < 0:
                x_k -= (x_k - (w//2))

            if y_k + (h//2) - 32 > 0 :
                y_k -= (y_k + (h//2) - 32)
            elif y_k - (h//2) < 0:
                y_k -= (y_k - (h//2))


            x[i,:,x_k - (w//2):x_k + (w//2), y_k - (h//2) : y_k + (h//2)] = original_x[rand_index[i],:,x_k - (w//2):x_k + (w//2), y_k - (h//2) : y_k + (h//2)]
        elif args.augmentation.endswith('v2'):
            Saliencyfmap = grayscale_vanilla_grads[rand_index[i],0,:,:]

            maximum_indices = np.unravel_index(np.argmax(Saliencyfmap, axis=None), Saliencyfmap.shape)
            second_x_k = maximum_indices[0]
            second_y_k = maximum_indices[1]
            
            if second_x_k + (w//2) - 32 > 0 :
                second_x_k -= (second_x_k + (w//2) - 32)
            elif second_x_k - (w//2) < 0:
                second_x_k -= (second_x_k - (w//2))

            if second_y_k + (h//2) - 32 > 0 :
                second_y_k -= (second_y_k + (h//2) - 32)
            elif second_y_k - (h//2) < 0:
                second_y_k -= (second_y_k - (h//2))

            Saliencyfmap = grayscale_vanilla_grads[i,0,:,:]

            maximum_indices = np.unravel_index(np.argmax(Saliencyfmap, axis=None), Saliencyfmap.shape)
            first_x_k = maximum_indices[0]
            first_y_k = maximum_indices[1]
            
            if first_x_k + (w//2) - 32 > 0 :
                first_x_k -= (first_x_k + (w//2) - 32)
            elif first_x_k - (w//2) < 0:
                first_x_k -= (first_x_k - (w//2))

            if first_y_k + (h//2) - 32 > 0 :
                first_y_k -= (first_y_k + (h//2) - 32)
            elif first_y_k - (h//2) < 0:
                first_y_k -= (first_y_k - (h//2))


            x[i,:,first_x_k - (w//2):first_x_k + (w//2), first_y_k - (h//2) : first_y_k + (h//2)] = original_x[rand_index[i],:,second_x_k - (w//2):second_x_k + (w//2), second_y_k - (h//2) : second_y_k + (h//2)]
    return x

# def FeatureMapSaliency(x, rand_index, lam, grayscale_vanilla_grads):
#     # X 64, 64, 32, 32
#     Saliencyfmap = grayscale_vanilla_grads[0]
#     maximum_indices = np.unravel_index(np.argmax(Saliencyfmap, axis=None), Saliencyfmap.shape)
#     x_k = maximum_indices[0]
#     y_k = maximum_indices[1]
    
#     if x_k + (lam//2) - 32 > 0 :
#         x_k -= (x_k + (lam//2) - 32)
#     elif x_k - (lam//2) < 0:
#         x_k -= (x_k - (lam//2))

#     if y_k + (lam//2) - 32 > 0 :
#         y_k -= (y_k + (lam//2) - 32)
#     elif y_k - (lam//2) < 0:
#         y_k -= (y_k - (lam//2))

#     print(x.shape)
#     x[:,:,x_k - lam:x_k + lam, y_k - lam : y_k + lam] = x[rand_index,:,x_k - lam:x_k + lam, y_k - lam : y_k + lam]
#     return x