import numpy as np

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def get_lam(bbx1, bbx2, bby1, bby2, distribution, rand_index):
    a_lam = 0
    b_lam = 0
    for idx in range(len(distribution)):
        o_w = int(distribution[idx][3] - distribution[idx][2])
        o_h = int(distribution[idx][1] - distribution[idx][0])
        
        c_w = int(distribution[rand_index[idx]][3] - distribution[rand_index[idx]][2])
        c_h = int(distribution[rand_index[idx]][1] - distribution[rand_index[idx]][0])

        if distribution[idx][2] < bbx2 and distribution[idx][3] > bbx1 and distribution[idx][0] < bby2 and distribution[idx][1] > bby1:
            start_x = max(distribution[idx][2], bbx1)
            end_x = min(distribution[idx][3], bbx2)

            start_y = max(distribution[idx][0], bby1)
            end_y = min(distribution[idx][1], bby2)

            w = end_x - start_x
            h = end_y - start_y
            a_lam += 1 - ((w*h)/(o_w*o_h))
        else:
            a_lam += 1

        if distribution[rand_index[idx]][2] < bbx2 and distribution[rand_index[idx]][3] > bbx1 and distribution[rand_index[idx]][0] < bby2 and distribution[rand_index[idx]][1] > bby1:
            start_x = max(distribution[rand_index[idx]][2], bbx1)
            end_x = min(distribution[rand_index[idx]][3], bbx2)

            start_y = max(distribution[rand_index[idx]][0], bby1)
            end_y = min(distribution[rand_index[idx]][1], bby2)

            w = end_x - start_x
            h = end_y - start_y
            b_lam += ((w*h)/(c_w*c_h))
        else:
            b_lam += 0
    
    if a_lam != 0 :
        a_lam /= len(distribution)
    if b_lam != 0 :
        b_lam /= len(distribution)
    
    return a_lam, b_lam


        

