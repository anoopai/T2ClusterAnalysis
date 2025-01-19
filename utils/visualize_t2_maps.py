def visualize_t2_maps(t2_map1_path, t2_map1_filt_path, t2_map2_path, t2_map2_filt_path, result_path):
    
    
    import dosma as dm
    import os
    import nibabel as nib

    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    t2_map1 = nib.load(t2_map1_path).get_fdata()  
    t2_map1_filt = nib.load(t2_map1_filt_path).get_fdata()
    t2_map2 = nib.load(t2_map2_path).get_fdata()
    t2_map2_filt = nib.load(t2_map2_filt_path).get_fdata()

    n= 30 # Choose a slice in the middle of the volume      

    # FIgure 2: Visualize the segmentations only
    plt.figure(figsize=(20, 7))

    ax1 = plt.subplot(1, 4, 1)
    im1 = ax1.imshow(t2_map1[:,:,n], cmap='viridis', alpha=1)
    plt.title('T2 map of Image1 : Unfiltered', fontsize=14)
    plt.xticks([])
    plt.yticks([])

    ax2 = plt.subplot(1, 4, 2)
    im2 = ax2.imshow(t2_map1_filt[:,:,n], cmap='viridis', alpha=1)
    plt.title('T2 Map of Image 1: Filtered', fontsize=14)
    plt.xticks([])
    plt.yticks([])
    
    ax3 = plt.subplot(1, 4, 3)
    im3 = ax3.imshow(t2_map2[:,:,n], cmap='viridis', alpha=1)
    plt.title('T2 map of Image2 : Unfiltered', fontsize=14)
    plt.xticks([])
    plt.yticks([])
    
    ax4 = plt.subplot(1, 4, 4)
    im4 = ax4.imshow(t2_map2_filt[:,:,n], cmap='viridis', alpha=1)
    plt.title('T2 Map of Image 2: Filtered', fontsize=14)
    plt.xticks([])
    plt.yticks([])
    

    plt.tight_layout()

    # plt_save_path= os.path.join(reg_check_path, f'{fixed_img_path.split(os.sep)[-2]}_slice{n}.jpg')
    plt_save_path= os.path.join(result_path, f'T2_maps_slice{n}.jpg')
    plt.savefig(plt_save_path)
    plt.close()