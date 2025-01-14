def apply_volume_threshold(difference_map_path, volume_threshold):
    
    '''
    map: 3D map path (.nii format)
    volume_threshold: threshold for the volume of the cluster.

    Returns: cluster_map_thresholded
    '''   
    
    import cc3d
    import numpy as np
    import nibabel as nib
    from utils.connected_component_analysis import connected_component_analysis
    
    
    difference_map = nib.load(difference_map_path)
    
    # connected component analysis to pick the clusters greater than the volume threshold
    difference_map_vol_thresholded= connected_component_analysis(difference_map.get_fdata(), threshold=volume_threshold)
    
    difference_map_vol_thresholded_mv = nib.Nifti1Image(difference_map_vol_thresholded, affine=difference_map.affine)
    
    return difference_map_vol_thresholded_mv