def get_values_per_t2c(cluster_map, fc_seg_common):
    
    '''
    Input:
    cluster_map: 3D numpy array with integer values representing different clusters
    fc_seg_common: 3D numpy array with integer values representing different clusters
    save_path: path to save the results
    
    Output:
    t2c_num_voxels: dictionary with cluster labels as keys and number of voxels as values
    t2c_percents: dictionary with cluster labels as keys and percentage of voxels as values
    t2c_sizes: dictionary with cluster labels as keys and size of the cluster as values
    t2c_means: dictionary with cluster labels as keys and mean of the cluster as values
    t2c_stds: dictionary with cluster labels as keys and standard deviation of the cluster as values
    t2c_medians: dictionary with cluster labels as keys and median of the cluster as values

    '''
    
    import os
    import cc3d
    import numpy as np
    import pandas as pd
    import nibabel as nib

    # Load baseline and target medical volumes (mv): 3D
    cluster_binary = np.where(~np.isnan(cluster_map.get_fdata()), 1, 0)
    
    fc_seg_common_binary = np.where(~np.isnan(fc_seg_common), 1, 0)
    
    # Find the number of voxels that are not non-nan from the difference map/feg common mask
    fc_voxels = np.sum(~np.isnan(fc_seg_common_binary))
    voxel_dims= cluster_map.header.get_zooms()
    
    # Get labels and other info for all the clusters using connected component analysis
    labels = cc3d.connected_components(cluster_binary, connectivity=6)
    all_voxels = cc3d.statistics(labels)['voxel_counts']
    
    # To identify and remove background label from the connected component analysis
    background_voxels = max(all_voxels)
    cluster_labels = np.where(all_voxels < background_voxels)[0]
    
    t2c_num_voxels= {}
    t2c_percents = {}
    t2c_sizes = {}
    t2c_means = {}
    t2c_stds = {}
    t2c_medians = {}
    
    if len(cluster_labels) > 0:
        
        for cluster_label in cluster_labels:
            cluster_mask = (labels == cluster_label)
            cluster_single = np.where(cluster_mask, cluster_map.get_fdata(), np.nan)
            
            if np.sum(cluster_mask) > 0:
                t2c_voxels= np.sum(cluster_mask)
                t2c_percent = (100 * np.sum(cluster_mask)) / fc_voxels
                t2c_size= t2c_voxels * voxel_dims[0] * voxel_dims[1] * voxel_dims[2]
                t2c_mean = np.nanmean(cluster_single)
                t2c_std = np.nanstd(cluster_single)
                t2c_median = np.nanmedian(cluster_single)
            else:
                t2c_voxels = np.sum(cluster_mask)
                t2c_percent = 0
                t2c_size = 0
                t2c_mean = 'NaN'
                t2c_std = 'NaN'
                t2c_median = 'NaN'           
            
            t2c_num_voxels[cluster_label] = t2c_voxels
            t2c_percents[cluster_label] = t2c_percent
            t2c_sizes[cluster_label] = t2c_size
            t2c_means[cluster_label] = t2c_mean
            t2c_stds[cluster_label] = t2c_std
            t2c_medians[cluster_label] = t2c_median
    
    return t2c_num_voxels, t2c_percents, t2c_sizes, t2c_means, t2c_stds, t2c_medians