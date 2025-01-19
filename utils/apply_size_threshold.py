def apply_size_threshold(difference_map_path, size_threshold, save_path):
    
    '''
    map: 3D map path (.nii format)
    siz_threshold: threshold for the sizeume of the cluster.

    Returns: cluster_map_thresholded
    '''   
    
    import cc3d
    import numpy as np
    import nibabel as nib
    from utils.connected_component_analysis import connected_component_analysis
    
    
    difference_map = nib.load(difference_map_path)
    
    # connected component analysis to pick the clusters greater than the size threshold
    difference_map_size_thresholded= connected_component_analysis(difference_map.get_fdata(), threshold=size_threshold)
    
    difference_map_size_thresholded_mv = nib.Nifti1Image(difference_map_size_thresholded, affine=difference_map.affine)
    
    nib.save(difference_map_size_thresholded_mv, save_path)
    
    return difference_map_size_thresholded_mv