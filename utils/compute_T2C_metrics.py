def compute_T2C_metrics(cluster_map_path, difference_map_path, save_path):
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
    
    if t2_voxels > 0: 
        t2c_percent = 100* (t2_voxels / fc_voxels)
        t2c_size= (t2_voxels * voxel_dims[0] * voxel_dims[1] * voxel_dims[2])/ len(cluster_labels)
        t2c_num = len(cluster_labels)
        t2c_mean = np.nanmean(cluster_map.get_fdata())
        t2c_std = np.nanstd(cluster_map.get_fdata())
        t2c_median = np.nanmedian(cluster_map.get_fdata())
    else:
        t2c_percent = 0
        t2c_size = 0
        t2c_num = 0
        t2c_mean = 'NaN'
        t2c_std = 'NaN'
        t2c_median = 'NaN'
    
    # Initialize the data with the total T2C metrics
    data_t2c = pd.DataFrame({
        'Region': 'FC all',
        'T2C Percent': [t2c_percent],
        'T2C Size (mm^3)': t2c_size,
        'T2C Num': t2c_num,
        'T2C Mean (ms)': [t2c_mean],
        'T2C Std (ms)': [t2c_std],
        'T2C Median (ms)': [t2c_median],        
        'T2C Voxels': [t2_voxels], 
        'Region Voxels': [fc_voxels],
    })
    
    # if len(cluster_labels) > 0:
    #     # Collect data rows in a list
    #     rows = []
        
    #     for cluster_label in cluster_labels:
    #         cluster_mask = (labels == cluster_label)
    #         cluster_single = np.where(cluster_mask, cluster_map.get_fdata(), np.nan)
            
    #         if np.sum(cluster_mask) > 0:
    #             t2c_voxels= np.sum(cluster_mask)
    #             t2c_percent = (100 * np.sum(cluster_mask)) / fc_voxels
    #             t2c_size= t2c_voxels * voxel_dims[0] * voxel_dims[1] * voxel_dims[2]
    #             t2c_mean = np.nanmean(cluster_single)
    #             t2c_std = np.nanstd(cluster_single)
    #             t2c_median = np.nanmedian(cluster_single)
    #         else:
    #             t2c_voxels = 0
    #             t2c_percent = 0
    #             t2c_size = 0
    #             t2c_mean = 'NaN'
    #             t2c_std = 'NaN'
    #             t2c_median = 'NaN'
            
    #         # Append a new row to the list
    #         rows.append({
    #             'T2C Label': cluster_label,
    #             'Region': 'FC all',
    #             'Region Label': 1 
    #             'T2C Voxels': t2c_voxels,
    #             'FC Voxels': fc_voxels,
    #             'T2C Percent': t2c_percent,
    #             'T2C Size (mm^3)': t2c_size,
    #             'T2C Num': 1,
    #             'T2C Mean (ms)': t2c_mean,
    #             'T2C Std (ms)': t2c_std,
    #             'T2C Median (ms)': t2c_median
    #         })
            
    #     # Create DataFrame from collected rows
    #     if rows:
    #         data_t2c_single = pd.DataFrame(rows)
    #         data_t2c = pd.concat([data_t2c, data_t2c_single], axis=0, ignore_index=True)
    #         data_t2c  = data_t2c.reset_index(drop=True)
        
    # Save the dataframe to an excel file
    append_df_to_excel(
        data= data_t2c, 
        sheet='T2C Metrics Region-wise', 
        save_path = save_path)
    
    return data_t2c