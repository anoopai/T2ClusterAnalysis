def compute_T2C_metrics(cluster_map_path, difference_map_path):
    '''
    cluster_map_path: path to the cluster map
    difference_map_path: path to the difference map. 
    NOTE: We use difference maps instead of the segmentation mask here to count the total volume of femoral cartilage,
    because the eroding (optional step) during T2 maps filtering removes some outer layers of voxels from the FC segmentation mask. 
    Using FC segmentation mask would result in a overestimating the volume of the cartilage (if erosion has been applied).
    '''
    
    import os
    import cc3d
    import numpy as np
    import pandas as pd
    import nibabel as nib
    from utils.append_df_to_excel import append_df_to_excel
    
    # Load baseline and target medical volumes (mv): 3D
    cluster_map = nib.load(cluster_map_path)
    cluster_binary = np.where(~np.isnan(cluster_map.get_fdata()), 1, 0)
    
    diff_map = nib.load(difference_map_path)
    diff_map_binary = np.where(~np.isnan(cluster_map.get_fdata()), 1, 0)
    
    # Find the number of voxels that are not non-nan from the difference map
    fc_voxels = np.sum(~np.isnan(diff_map.get_fdata()))
    t2_voxels = np.sum(cluster_binary)

    voxel_dims= cluster_map.header.get_zooms()
    
    # Get labels and other info for all the clusters using connected component analysis
    labels = cc3d.connected_components(cluster_binary, connectivity=6)
    all_voxels = cc3d.statistics(labels)['voxel_counts']
    
    # To identify and remove background label from the connected component analysis
    background_voxels = max(all_voxels)
    cluster_labels = np.where(all_voxels < background_voxels)[0]
    
    # Initialize the data with the total T2C metrics
    data_t2c = pd.DataFrame({
        'T2C Label': 'All', 
        'T2C Voxels': [t2_voxels],
        'FC Voxels': [fc_voxels],
        'T2C Percent': [(t2_voxels / fc_voxels) * 100],
        'T2C Size (mm^3)': (t2_voxels * voxel_dims[0] * voxel_dims[1] * voxel_dims[2])/ len(cluster_labels),
        'T2C Num': len(cluster_labels),
        'T2C Mean (ms)': [np.nanmean(cluster_map.get_fdata())],
        'T2C Std (ms)': [np.nanstd(cluster_map.get_fdata())],
        'T2C Median (ms)': [np.nanmedian(cluster_map.get_fdata())]
    })
    
    # Collect data rows in a list
    rows = []
    
    for cluster_label in cluster_labels:
        cluster_mask = (labels == cluster_label)
        cluster_single = np.where(cluster_mask, cluster_map.get_fdata(), np.nan)
        
        # Append a new row to the list
        rows.append({
            'T2C Label': str(cluster_label),
            'T2C Voxels': np.sum(cluster_mask),
            'FC Voxels': fc_voxels,
            'T2C Percent': (100 * np.sum(cluster_mask)) / fc_voxels,
            'T2C Size (mm^3)': np.sum(cluster_mask) * voxel_dims[0] * voxel_dims[1] * voxel_dims[2],
            'T2C Num': 1,
            'T2C Mean (ms)': np.nanmean(cluster_single),
            'T2C Std (ms)': np.nanstd(cluster_single),
            'T2C Median (ms)': np.nanmedian(cluster_single)
        })
        
    
    # Create DataFrame from collected rows
    if rows:
        data_t2c_single = pd.DataFrame(rows)
        data_t2c = pd.concat([data_t2c, data_t2c_single], axis=0, ignore_index=True)
    
    return data_t2c