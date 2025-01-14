def connected_component_analysis(cluster, threshold):
    
    import numpy as np
    import cc3d
    import copy

    # Convert image to binary
    
    cluster_binary= np.where(~np.isnan(cluster), 1, 0)

    # connected component analysis of the difference map. Removes dust with voxels lesser than threshold
    labels=  cc3d.dust(cluster_binary.astype(np.uint64), connectivity=6, threshold=threshold, in_place=False)
    
    # Compute number of voxels for each label
    all_voxels= cc3d.statistics(labels)['voxel_counts']
    background_voxels = max(all_voxels)
    
    # get label number of the mask label
    large_cluster_label= np.where(all_voxels < background_voxels)[0]

    clusters_selected=np.empty(0)

    # Only selected clusters as values and zero for dust and background
    clusters_selected= cluster * np.where(labels==large_cluster_label, 1, np.nan)

    return clusters_selected