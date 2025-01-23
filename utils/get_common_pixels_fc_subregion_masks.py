def get_common_pixels_fc_subregion_masks(seg1_path, seg2_path, save_path):
        
    '''
    baseline_qmap: baseline quantitatrive map (T2 and T1rho)
    followup_qmap: followup baseline quantitatrive map (T2 and T1rho)
    baseline_mask/followup_mask: Mask for the tissue you want to use for baseline and followup qmaps (will be same if baseline and followup qmaps are registered)
    diff_map_save_path: path to save the difference map
    
    '''
    import numpy as np
    import nibabel as nib

    # load the qmaps and masks
    seg1 = nib.load(seg1_path)
    seg2 = nib.load(seg2_path) 
    
    seg1_data = seg1.get_fdata()
    seg2_data = seg2.get_fdata()
        
    # Assume baseline and target are your input arrays
    seg1_matched = np.where(np.isnan(seg2_data), np.nan, seg1_data)
    seg2_matched = np.where(np.isnan(seg1_data), np.nan, seg2_data)

    # Convert the numpy array to a Nifti1Image
    seg = nib.Nifti1Image(seg1_matched, seg1.affine)
    
    # Save the Nifti1Image
    nib.save(seg, save_path)