def compute_DSC(mask1, mask2):

    import os
    import numpy as np
    import matplotlib.pyplot as plt
        
    mask1_binary = np.where(mask1 > 0, 1, 0)
    mask2_binary = np.where(mask2 > 0, 1, 0)

    intersection = np.logical_and(mask1_binary, mask2_binary)
    dice_score = 2.0 * intersection.sum() / (mask1_binary.sum() + mask2_binary.sum())
    
    return dice_score