def visualize_sgd_registration(fixed_img_path, moving_img_path, moving_img_reg_path, fixed_mask_path, moving_mask_path, moving_mask_reg_path, result_path):
    
    
    import dosma as dm
    import os
    import nibabel as nib

    import numpy as np
    import matplotlib.pyplot as plt
    import sigpy as sp
    import traceback
    import copy
    import shutil
    from pathlib import Path

    from dosma.scan_sequences import QDess

    # Load qdess scan
    qdess_fixed = QDess.load(fixed_img_path).volumes[0].A
    qdess_moving = QDess.load(moving_img_path).volumes[0].A
    qdess_moving_reg= QDess.load(moving_img_reg_path).volumes[0].A

    seg_fixed = nib.load(fixed_mask_path).get_fdata()  
    seg_moving = nib.load(moving_mask_path).get_fdata()
    seg_moving_reg= nib.load(moving_mask_reg_path).get_fdata()

    seg_fixed_nan= np.where(seg_fixed==0, np.nan, 1)
    seg_moving_nan= np.where(seg_moving==0, np.nan, 1)
    seg_moving_reg_nan= np.where(seg_moving_reg==0, np.nan, 1)

    # Figure 1: Visualize the registered image with segmentation
    # Create figure and axes objects with specified layout and size
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 7)) # Adjust figsize as needed
    
    n= 30 # Choose a slice in the middle of the volume
    
    # Plot the first image with its segmentation
    axes[0].imshow(qdess_fixed[:, :, n], cmap='Purples_r', alpha=1)
    axes[0].imshow(seg_fixed_nan[:, :, n], cmap="Reds_r", alpha=1)  # Adjust alpha to make segmentation visible over scan
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_title("Image1 Scan and FC segmentation", fontsize=14)

    # Plot the second image with its segmentation
    axes[1].imshow(qdess_moving[:, :, n], cmap='Greens_r', alpha=1)
    axes[1].imshow(seg_moving_nan[:, :, n], cmap="Greys_r", alpha=1)  # Adjust alpha to make segmentation visible over scan
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_title("Image2 Scan and FC segmentation", fontsize=14)

    # Show the registered image overlaid with the initial image
    axes[2].imshow(qdess_fixed[:, :, n], cmap='Purples', alpha=1)
    axes[2].imshow(qdess_moving_reg[:, :, n], cmap="Greens", alpha=0.4)
    axes[2].imshow(seg_fixed_nan[:, :, n], cmap="Reds_r", alpha=0.7)  # Adjust alpha to make segmentation visible over scan
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[2].set_title("Signed distance field based registration of \n Image 2 to Image 1", fontsize=14)

    # Ensure layout is tight so everything fits nicely
    plt.tight_layout()
    plt.show()

    # plt_save_path= os.path.join(reg_check_path, f'{fixed_img_path.split(os.sep)[-2]}_slice{n}.jpg')
    plt_save_path= os.path.join(result_path, f'Reg_qdess_slice{n}.jpg')
    plt.savefig(plt_save_path)
    plt.close()
            

    # FIgure 2: Visualize the segmentations only
    plt.figure(figsize=(15, 7))

    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.imshow(seg_fixed_nan[:, :, n], cmap='Set1', alpha=1)
    plt.title('Femoral Cartilage mask \n for Image 1 (fixed scan)', fontsize=14)
    plt.xticks([])
    plt.yticks([])

    ax2 = plt.subplot(1, 3, 2)
    im2 = ax2.imshow(seg_moving_reg_nan[:, :, n], cmap='Set2', alpha=1)
    plt.title('Femoral Cartilage re-segmented \n post registartion for Image 2 (moving scan)', fontsize=14)
    plt.xticks([])
    plt.yticks([])

    ax3 = plt.subplot(1, 3, 3)
    im3 = ax3.imshow(seg_fixed_nan[:, :, n], cmap='Set1', alpha=0.5)
    im3 = ax3.imshow(seg_moving_reg_nan[:, :, n], cmap='Set2', alpha=0.5)
    plt.title('Overlap of the two femoral cartilage masks', fontsize=14)
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()

    # plt_save_path= os.path.join(reg_check_path, f'{fixed_img_path.split(os.sep)[-2]}_slice{n}.jpg')
    plt_save_path= os.path.join(result_path, f'Reg_seg_fc_slice{n}.jpg')
    plt.savefig(plt_save_path)
    plt.close()