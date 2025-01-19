def apply_intensity_threshold (difference_map_path, intensity_threshold, cluster_type, save_path):
    
    '''
    difference_map: 3D difference map path (.nii format)
    intensity_threshold: threshold for the intensity of the cluster.
    cluster_type: pos or neg
    pos = takes t2 values greater than the threshold
    neg = takes t2 values less than the threshold 
    
    '''
    import numpy as np
    import nibabel as nib

    difference_map = nib.load(difference_map_path)

    if cluster_type == 'pos':
        diff_maps_intensity_thresh= np.where(difference_map.get_fdata() <= intensity_threshold, np.nan, difference_map.get_fdata())
    elif cluster_type == 'neg':
        diff_maps_intensity_thresh= np.where(difference_map.get_fdata() >= -intensity_threshold, np.nan, difference_map.get_fdata())
    else:
        print('Please enter the correct cluster type: pos or neg')
        return None
    
    diff_maps_intensity_thresh_mv = nib.Nifti1Image(diff_maps_intensity_thresh, affine=difference_map.affine)
    
    nib.save(diff_maps_intensity_thresh_mv, save_path)

    return diff_maps_intensity_thresh_mv