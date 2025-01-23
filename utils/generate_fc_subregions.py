def generate_fc_subregions(input_seg_file_path, output_seg_file_path):
    
    # Note: This works only with pymskt environment
    
    '''
    This function generates the segmentation mask of femoral cartilage subregions (Anterior, medial and lateral weight-bearing, medial and lateral posterior).
    Input (str): input_seg_file_path: path to the segmentation mask file of femur, tibia, and femoral cartilage (.nii format)
    Output (SimpleITK image): fc_regions_img: segmentation mask of femoral cartilage subregions (femoral cartilage, weightbearing, trochlea, posterior condyles) in the format used by Dosma/ other data
    
    '''
    
    import os
    import numpy as np
    import nibabel as nib
    import pandas as pd
    import SimpleITK as sitk
    from pymskt.image.cartilage_processing import get_knee_segmentation_with_femur_subregions

    from utils.realign_seg_masks_SITK_format import realign_seg_masks_SITK_format
    from utils.realign_seg_masks_dosma_format import realign_seg_masks_dosma_format

    # realign the segmentation mask to the format used by SimpleITK
    seg_reorient = realign_seg_masks_SITK_format(input_seg_file_path)
    # seg_reorient = sitk.ReadImage(input_seg_file_path)

    fc_regions = get_knee_segmentation_with_femur_subregions(
        seg_image= seg_reorient,
        fem_cart_label_idx=2,
        wb_region_percent_dist=0.6,
        # femur_label=1,
        med_tibia_label=3,
        lat_tibia_label=4,
        ant_femur_mask=11,
        med_wb_femur_mask=12,
        lat_wb_femur_mask=13,
        med_post_femur_mask=14,
        lat_post_femur_mask=15,
        verify_med_lat_tib_cart=True,
        tibia_label=8,
        ml_axis=0,
    )
    
    # realign the segmentation mask to the format used by Dosma/ other data
    fc_regions_dosma = realign_seg_masks_dosma_format(fc_regions)
    # fc_regions_dosma= fc_regions

    # remove non FC labels (Femur, Tibia, Meniscus, PC, TC, etc.)
    # Extract the array from the SimpleITK image
    fc_regions_array = sitk.GetArrayFromImage(fc_regions_dosma)

    # Create an array of zeros with the same shape as the input image
    fc_regions_array[fc_regions_array < 10] = 0
    
    # check if there are 6 labels (5 subregions and 1 background)
    if len(np.unique(fc_regions_array)) != 6:
        print(f"Expected 6 unique labels in the segmentation mask, but found {len(np.unique(fc_regions_array))}.")

    # Create a new SimpleITK image from the zero array
    fc_regions_img = sitk.GetImageFromArray(fc_regions_array)

    # Copy the original image's metadata (e.g., spacing, origin, direction)
    fc_regions_img.SetSpacing(fc_regions_dosma.GetSpacing())
    fc_regions_img.SetOrigin(fc_regions_dosma.GetOrigin())
    fc_regions_img.SetDirection(fc_regions_dosma.GetDirection())
    fc_regions_img = sitk.Cast(fc_regions_img, sitk.sitkInt16)
    
    print(f"Saving femoral cartilage subregions segmentation mask to: {os.path.basename(output_seg_file_path)}")
    sitk.WriteImage(fc_regions_img, output_seg_file_path)
    
    return fc_regions_img
