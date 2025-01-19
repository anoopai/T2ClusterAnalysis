def visualize_t2_cluster_analysis(t2_difference_map_path, t2_int_threshold_path, t2_size_threshold_path, result_path):
    
    
    import dosma as dm
    import os
    import nibabel as nib

    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    t2_difference_map = nib.load(t2_difference_map_path).get_fdata()
    t2_int_threshold= nib.load(t2_int_threshold_path).get_fdata()
    t2_size_threshold= nib.load(t2_size_threshold_path).get_fdata()
    

    n= 30 # Choose a slice in the middle of the volume      

    # Figure 2: Visualize the segmentations only
    plt.figure(figsize=(16, 6))

    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.imshow(t2_difference_map[:,:,n], cmap='bwr', alpha=1)
    plt.title('T2 Difference Map', fontsize=14)
    plt.xticks([])
    plt.yticks([])

    ax2 = plt.subplot(1, 3, 2)
    im2 = ax2.imshow(t2_int_threshold[:,:,n], cmap='bwr', alpha=1)
    plt.title('T2 Difference Map: Intensity Thresholded', fontsize=14)
    plt.xticks([])
    plt.yticks([])
    
    ax3 = plt.subplot(1, 3, 3)
    im3 = ax3.imshow(t2_size_threshold[:,:,n], cmap='bwr', alpha=1)
    plt.title('T2 Difference Map: Size Thresholded', fontsize=14)
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()

    # plt_save_path= os.path.join(reg_check_path, f'{fixed_img_path.split(os.sep)[-2]}_slice{n}.jpg')
    plt_save_path= os.path.join(result_path, f'T2_cluster_analysis_slice{n}.jpg')
    plt.savefig(plt_save_path)
    plt.close()