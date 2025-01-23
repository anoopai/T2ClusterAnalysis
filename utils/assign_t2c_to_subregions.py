def assign_t2c_to_subregions(cluster_map_path, fc_subregions_path, t2c_save_path, t2c_results_save_path):
    
    import numpy as np
    import cc3d
    import nibabel as nib
    import pandas as pd
    
    from utils.com_of_labeled_mask import com_of_labeled_mask
    from utils.eucledian_distance_t2c_subregions import eucledian_distance_t2c_subregions
    from utils.t2c_to_subregion_map import t2c_to_subregion_map 
    from utils.get_num_voxels import get_num_voxels
    from utils.get_values_per_t2c import get_values_per_t2c
    from utils.append_df_to_excel import append_df_to_excel
    

    # Load cluster map and convert to binary    
    cluster_map_img = nib.load(cluster_map_path)
    cluster_map= cluster_map_img.get_fdata()
    affine = nib.load(cluster_map_path).affine
    
    voxel_dims= cluster_map_img.header.get_zooms()
    
    fc_subregions_mask = nib.load(fc_subregions_path).get_fdata().astype(int)
    
    # get COM for individual clusters
    cluster_binary= np.where(~np.isnan(cluster_map), 1, 0) # convert to binary
    cc_labels = cc3d.connected_components(cluster_binary, connectivity = 6) # connected component analysis

    # Compute number of voxels for each label
    all_voxels= cc3d.statistics(cc_labels)['voxel_counts'] # get the number of voxels for each label
    background_voxels = max(all_voxels) # get the number of voxels in the background
    labels_clusters= np.where(all_voxels < background_voxels)[0] # get the labels of all clusters without the background

    # Create a mask to keep only the specified labels
    mask = np.isin(cc_labels, labels_clusters)
    
    # Apply the mask to the connected component labels
    t2_cluster_label_data = np.where(mask, cc_labels, 0)
    
    # Compute the center of mass for each cluster
    cluster_com_dict= com_of_labeled_mask(t2_cluster_label_data)
    
    # Compute the center of mass values for fc subregions
    subregions_com_dict= com_of_labeled_mask(fc_subregions_mask) 
    
    # Compute the distance between the COM of the T2C and the COM of the subregions
    cluster_subregion_distances_dict = eucledian_distance_t2c_subregions(
        data1_dict= subregions_com_dict, 
        data2_dict= cluster_com_dict
        )
        
    # Assign T2 cluster to subregions
    t2c_in_subregion_dict= t2c_to_subregion_map(cluster_subregion_distances_dict)
    
    ############# Compute number of voxels for each label ##############
    cluster_num_voxels_dict = get_num_voxels(cc_labels) # number of voxels per cluster
    subregion_num_voxels_dict = get_num_voxels(fc_subregions_mask)
    
    ########## Get T2 mean, std, median for each cluster ##############
    
    _, cluster_percent_dict, cluster_size_dict, cluster_t2mean_dict, \
            cluster_t2std_dict, cluster_t2median_dict= get_values_per_t2c(cluster_map_img, fc_subregions_mask)
    
    ########## Convert all the dictionaries to a dataframes #############
    ########### T2C COMs ###########
    cluster_com_data=pd.DataFrame.from_dict(cluster_com_dict, orient= 'index', columns=['X-value', 'Y-value', 'Z-Value'])
    cluster_com_data=cluster_com_data.reset_index().rename(columns={'index': 'T2C'})
    
    ########### FC sub-region COM ###########
    subregions_com_data = pd.DataFrame.from_dict(subregions_com_dict, orient= 'index', columns=['X-value', 'Y-value', 'Z-value'])
    subregions_com_data = subregions_com_data.reset_index().rename(columns={'index': 'Region Label'})

    subregion_mapping = {
        11: "AN",
        12: "MC",
        13: "LC",
        14: "MP",
        15: "LP"
    }

    # Insert the subregion column
    subregions_com_data['Region'] = subregions_com_data['Region Label'].map(subregion_mapping)
    subregions_com_data.insert(1, 'Region', subregions_com_data.pop('Region'))
    
    ########## Distance of each T2C COM to each subregion COM ###########
    t2c_subregion_distances_all_data=pd.DataFrame.from_dict(cluster_subregion_distances_dict, orient= 'index')
    t2c_subregion_distances_all_data=t2c_subregion_distances_all_data.reset_index().rename(columns={'index': 'T2C Label'})
    
    ###################### T2C in Subregions ##########################
    t2c_in_subregion_data=pd.DataFrame.from_dict(t2c_in_subregion_dict, orient= 'index', columns=['Region Label'])
    t2c_in_subregion_data=t2c_in_subregion_data.reset_index().rename(columns={'index': 'T2C Label'})
    t2c_in_subregion_data['Region'] = t2c_in_subregion_data['Region Label'].map(subregion_mapping)
    t2c_in_subregion_data.insert(1, 'Region', t2c_in_subregion_data.pop('Region'))
    t2c_in_subregion_data['Region Voxels'] = t2c_in_subregion_data['Region Label'].map(subregion_num_voxels_dict)
    t2c_in_subregion_data['T2C Voxels'] = t2c_in_subregion_data['T2C Label'].map(cluster_num_voxels_dict) 
    t2c_in_subregion_data['T2C Percent'] = (t2c_in_subregion_data['T2C Voxels'] / t2c_in_subregion_data['Region Voxels']) * 100
    t2c_in_subregion_data['T2C Size (mm^3)'] = t2c_in_subregion_data['T2C Label'].map(cluster_size_dict)
    t2c_in_subregion_data['T2C Num'] = 1
    t2c_in_subregion_data['T2C Mean (ms)'] = t2c_in_subregion_data['T2C Label'].map(cluster_t2mean_dict)
    t2c_in_subregion_data['T2C Std (ms)'] = t2c_in_subregion_data['T2C Label'].map(cluster_t2std_dict)
    t2c_in_subregion_data['T2C Median (ms)'] = t2c_in_subregion_data['T2C Label'].map(cluster_t2median_dict)
    t2c_in_subregion_data = t2c_in_subregion_data[['T2C Label', 'Region', 'T2C Percent', 'T2C Size (mm^3)', 'T2C Num', 'T2C Mean (ms)', 'T2C Std (ms)', 'T2C Median (ms)','T2C Voxels', 'Region Voxels']]

    # Assign sub-region values to T2C based on which region it belongs
    t2c_as_subregion_labels = np.zeros_like(cc_labels)
    for t2c_label, subregion_label in t2c_in_subregion_dict.items():
        t2c_as_subregion_labels[cc_labels == t2c_label] = subregion_label
        
    t2c_as_subregion_labels_img= nib.Nifti1Image(t2c_as_subregion_labels, affine)
    nib.save(t2c_as_subregion_labels_img, t2c_save_path)
    
    # append_df_to_excel(data=cluster_com_data, sheet= 'T2C COM', save_path= t2c_results_save_path)      
    # append_df_to_excel(data=subregions_com_data, sheet= 'Regions COM', save_path= t2c_results_save_path)
    # append_df_to_excel(data=t2c_subregion_distances_all_data, sheet= 'Distance bw T2C-Subregions COM', save_path= t2c_results_save_path)      
    append_df_to_excel(data=t2c_in_subregion_data, sheet= 'T2C Metrics Cluster-wise', save_path= t2c_results_save_path)

    return t2c_in_subregion_data                                          
    