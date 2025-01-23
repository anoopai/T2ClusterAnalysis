def sgd_based_registration(fixed_img_path, moving_img_path, moving_img_save_path, fixed_mask_path, moving_mask_path, elastix_file_path, reg_path):

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

    # check if the path is correct
    if os.path.exists(fixed_img_path) and os.path.exists(moving_img_path):

        # Delete anything in the output directory.
        # This is to make sure there aren't any lingering files that cause the run to succeed sometimes.
        shutil.rmtree(moving_img_save_path, ignore_errors=True)

        # Load qdess scan
        qdess_fixed = QDess.load(fixed_img_path)
        qdess_moving = QDess.load(moving_img_path)

        # Get echos of the qdess
        qdess_fixed_echo1 = qdess_fixed.volumes[0]
        qdess_moving_echo1 = qdess_moving.volumes[0]
        qdess_moving_echo2 = qdess_moving.volumes[1]

        # load the segmentation for femoral cartilage
        reader = dm.NiftiReader()
        seg_fixed = reader.load(fixed_mask_path)
        seg_moving = reader.load(moving_mask_path)

        seg_signed_fixed= signed_maurer_distance_map_image_filter(seg_fixed.A)
        seg_signed_fixed= seg_signed_fixed.astype(np.float64)
        seg_signed_threshold_0= np.where(seg_signed_fixed > 5, 1, seg_signed_fixed)
        seg_signed_mv_0 = dm.MedicalVolume(seg_signed_threshold_0, qdess_fixed_echo1.affine)

        seg_signed_moving= signed_maurer_distance_map_image_filter(seg_moving.A)
        seg_signed_moving= seg_signed_moving.astype(np.float64)
        seg_signed_threshold_1= np.where(seg_signed_moving > 5, 1, seg_signed_moving)
        seg_signed_mv_1 = dm.MedicalVolume(seg_signed_threshold_1, qdess_moving_echo1.affine)

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
        reg_echo1 = dm.apply_warp(qdess_moving_echo1, out_registration=registration_parameters)
        reg_echo2 = dm.apply_warp(qdess_moving_echo2, out_registration=registration_parameters)

        # Recontrust the qdess scans from the registered qdess echos
        qdess_moving_reg = QDess([reg_echo1, reg_echo2]) # make a qdess \

        # save qdess scans
        print(f'Saving registered qdess files')
        qdess_moving_reg.save(moving_img_save_path, save_custom=True, image_data_format=ImageDataFormat.nifti)
        
        # remove the output directory
        output_path= os.path.join(reg_path, f'registration_files')
        shutil.rmtree(output_path, ignore_errors=True)
      
    else:
        print("Please check the paths of inout images.")