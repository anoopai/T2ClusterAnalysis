def sgd_based_bilateral_qdess_registration(fixed_img_path, moving_img_path, moving_img_save_path, fixed_mask_path, moving_mask_path, elastix_file_path, reg_path, reg_check):

    '''
    This function performs registration between two images using signed distance map of the segmentation masks.
    
    fixed_img_path: str (path to the fixed image)
    moving_img_path: str (path to the moving image)
    moving_img_save_path: str (path to save the registered moving image)
    fixed_mask_path: str (path to the fixed mask)
    moving_mask_path: str (path to the moving mask)
    elastix_file_path: str (path to the elastix parameter file)
    reg_check: bool (if True, it will save the a jpg pciture of a random slice (n) of the segmentation mask of the fixed image overlayed on the registered moving image)
    '''
    
    # Import libraries and dependencies
    import dosma as dm
    import os

    import numpy as np
    import matplotlib.pyplot as plt
    import sigpy as sp
    import traceback
    import copy
    import shutil
    import nibabel as nib   
    from pathlib import Path

    from dosma import preferences
    from dosma.scan_sequences import QDess, CubeQuant, Cones
    from sigpy.plot import ImagePlot
    from dosma import ImageDataFormat
    from tkinter import font

    from utils.signed_maurer_distance_map_image_filter import signed_maurer_distance_map_image_filter

    output_path= os.path.join(reg_path, f'registration_files')
    
    shutil.rmtree(output_path, ignore_errors=True)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Delete anything in the output directory.
    # This is to make sure there aren't any lingering files that cause the run to succeed sometimes.
    shutil.rmtree(moving_img_save_path, ignore_errors=True)
    
    # Load qdess scan
    qdess_fixed = QDess.load(fixed_img_path)
    qdess_moving = QDess.load(moving_img_path)

    # Get echos of the qdess
    qdess_fixed_echo1 = qdess_fixed.volumes[0]
    qdess_fixed_echo2 = qdess_fixed.volumes[1]  
    qdess_moving_echo1 = qdess_moving.volumes[0]
    qdess_moving_echo2 = qdess_moving.volumes[1]
    
    # reformat the medical volumes (so that both legs are either right or left side)
    qdess_fixed_ornt = qdess_fixed_echo1.orientation

    # Flip the orientation of the moving qdess scans
    flipped_ornt= tuple('LR' if x == 'RL' else 'RL' if x == 'LR' else x for x in qdess_fixed_ornt)

    # Reformat the moving qdess scans
    qdess_moving_echo1_reornt= qdess_moving_echo1.reformat(flipped_ornt)
    qdess_moving_echo2_reornt= qdess_moving_echo2.reformat(flipped_ornt)
    qdess_moving_echo1_reornt= dm.MedicalVolume(qdess_moving_echo1_reornt.A, qdess_fixed_echo1.affine)
    qdess_moving_echo2_reornt= dm.MedicalVolume(qdess_moving_echo2_reornt.A, qdess_fixed_echo1.affine)
    
    # load the segmentation for femoral cartilage
    reader = dm.NiftiReader()
    seg_fixed = reader.load(fixed_mask_path)
    seg_moving = reader.load(moving_mask_path)
    seg_moving_reornt = seg_moving.reformat(flipped_ornt)
    seg_moving_reornt_mv= dm.MedicalVolume(seg_moving_reornt.A, seg_fixed.affine)  
    
    # Combine to form a Qdess Scan
    # qdess_moving_reornt = QDess([echo1_reornt, echo1_reornt])

    seg_fixed_binary= np.where(seg_fixed.A > 0, 1, seg_fixed.A)
    seg_moving_binary= np.where(seg_moving_reornt_mv.A > 0, 1, seg_moving_reornt_mv.A)

    seg_signed_fixed= signed_maurer_distance_map_image_filter(seg_fixed_binary)
    seg_signed_fixed= seg_signed_fixed.astype(np.float64)
    seg_signed_threshold_0= np.where(seg_signed_fixed > 5, 1, seg_signed_fixed)
    seg_signed_mv_0 = dm.MedicalVolume(seg_signed_threshold_0, qdess_fixed_echo1.affine)

    seg_signed_moving= signed_maurer_distance_map_image_filter(seg_moving_binary)
    seg_signed_moving= seg_signed_moving.astype(np.float64)
    seg_signed_threshold_1= np.where(seg_signed_moving > 5, 1, seg_signed_moving)
    seg_signed_mv_1 = dm.MedicalVolume(seg_signed_threshold_1, qdess_moving_echo1_reornt.affine)
    
    # Registration
    print("Starting Registration")

    output= dm.register(
        target=seg_signed_mv_0, 
        moving=seg_signed_mv_1, 
        parameters=elastix_file_path, 
        output_path=output_path, 
        return_volumes=True, 
        num_workers= 50,
        num_threads= 50,
        show_pbar=True)

    print("Registration completed")

    # gather the registration parameters
    registration_parameters = output["outputs"][0]
    registered_mask_1 = output['volume'][0]
    # registered_mask_1.save_volume(seg_reg_SDF_save_path)

    # Warp the Qdess scan using the registration parametrs obtained in the mask registration
    reg_echo1 = dm.apply_warp(qdess_moving_echo1_reornt, out_registration=registration_parameters)
    reg_echo2 = dm.apply_warp(qdess_moving_echo2_reornt, out_registration=registration_parameters)

    # Recontrust the qdess scans from the registered qdess echos
    qdess_moving_reg = QDess([reg_echo1, reg_echo2]) # make a qdess \  

    # save qdess scans
    print(f'Saving registered qdess files for {os.path.basename(moving_img_path)}')
    qdess_moving_reg.save(moving_img_save_path, save_custom=True, image_data_format=ImageDataFormat.nifti)
    
    # Save individual echos seperately (some issue with DOSMA save)
    nib_reg_echo1 = nib.Nifti1Image(reg_echo1.A, reg_echo1.affine)
    moving_img_save_echo1_path= os.path.join(moving_img_save_path, 'volumes/echo-000.nii.gz')
    if os.path.exists(moving_img_save_echo1_path):
        os.remove(moving_img_save_echo1_path)
    nib.save(nib_reg_echo1, moving_img_save_echo1_path)
    nib_reg_echo2 = nib.Nifti1Image(reg_echo2.A, reg_echo2.affine)
    moving_img_save_echo2_path= os.path.join(moving_img_save_path, 'volumes/echo-001.nii.gz')
    if os.path.exists(moving_img_save_echo2_path):
        os.remove(moving_img_save_echo2_path)
    nib.save(nib_reg_echo2, moving_img_save_echo2_path)

    # seg_fixed.save_volume(seg_reg_1_save_path)
    # if os.path.exists(seg_reg_0_save_path)== False:
    #     seg_fixed.save_volume(seg_reg_0_save_path)
        
    if reg_check:
        
        reg_check_path = os.path.join(reg_path, f'registration_check')

        if not os.path.exists(reg_check_path):
            os.makedirs(reg_check_path)             
    
        # Plot and save registrations for checking
        n= 30
        _, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,20), constrained_layout=False)
        axes[0].imshow(qdess_fixed.volumes[0].A[:,:,n], cmap='gray', alpha=1)
        axes[0].imshow(qdess_moving_reg.volumes[0].A[:,:,n], cmap="Blues", alpha= 0.7)
        axes[1].imshow(qdess_moving_reg.volumes[0].A[:,:,n], cmap='gray', alpha=1)
        axes[1].imshow(seg_fixed.A[:,:,n], cmap="Blues", alpha=0.7)
        axes[0].set_title("Fixed vs Moving Qdess Scan", fontsize=12)
        axes[1].set_title("Fixed Mask on Moving Qdess Scan", fontsize=12)

        plt_save_path= os.path.join(reg_check_path, f'{moving_img_path.split(os.sep)[-2]}_slice{n}.jpg')
        plt.savefig(plt_save_path)