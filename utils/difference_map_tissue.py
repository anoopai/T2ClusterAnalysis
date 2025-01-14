def difference_map_tissue(baseline_qmap_path, followup_qmap_path, baseline_mask_path, followup_mask_path, mask_erode=False, erode_size=1):
        
    '''
    baseline_qmap: baseline quantitatrive map (T2 and T1rho)
    followup_qmap: followup baseline quantitatrive map (T2 and T1rho)
    baseline_mask/followup_mask: Mask for the tissue you want to use for baseline and followup qmaps (will be same if baseline and followup qmaps are registered)
    diff_map_save_path: path to save the difference map
    
    '''
    import numpy as np
    import nibabel as nib
    from nibabel import Nifti1Image

    from utils.erode import erode
    
    # load the qmaps and masks
    baseline_qmap = nib.load(baseline_qmap_path)
    followup_qmap = nib.load(followup_qmap_path)
    baseline_mask = nib.load(baseline_mask_path)
    followup_mask = nib.load(followup_mask_path)
    
    
    # convert to binary mask (1s and 0s, if labels are not 1)
    baseline_mask_binary = np.where(baseline_mask.get_fdata() > 0, 1, 0)
    followup_mask_binary = np.where(followup_mask.get_fdata() > 0, 1, 0)
    
    if mask_erode== True:
        # erode the mask
        baseline_mask_eroded= erode(baseline_mask_binary, size=erode_size)
        followup_mask_eroded= erode(followup_mask_binary, size=erode_size)

        # apply the mask
        baseline_nan = np.where(baseline_mask_eroded == 1, baseline_qmap.get_fdata(), np.nan)
        followup_nan = np.where(followup_mask_eroded == 1, followup_qmap.get_fdata(), np.nan)
    
    else:
        baseline_nan = np.where(baseline_mask.get_fdata() == 1, baseline_qmap.get_fdata(), np.nan)
        followup_nan = np.where(followup_mask.get_fdata() == 1, followup_qmap.get_fdata(), np.nan)
        
    # Assume baseline and target are your input arrays
    baseline_matched = np.where(np.isnan(followup_nan), np.nan, baseline_nan)
    followup_matched = np.where(np.isnan(baseline_nan), np.nan, followup_nan)

    # Perform the subtraction
    diff_map = followup_matched - baseline_matched
    
    # Convert the numpy array to a Nifti1Image
    diff_map_nii = Nifti1Image(diff_map, followup_qmap.affine)

    return diff_map_nii