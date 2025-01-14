def t2c_to_subregion_map(t2c_subegion_distance_dict):
    
    '''
    Input: t2c_subegion_distance_dict: nested dictionary with keys as cluster labels and values as nested dictionary with keys as subregion labels and values as distance between the cluster COM and subregion COM
    
    Output = cluster_in_subregion: dictionary with keys as cluster labels and values as subregion labels
    '''
    
    import numpy as np
    
    cluster_in_subregion = {}
    
    for cluster_key, subregion_distances in t2c_subegion_distance_dict.items():
        
        subregion_keys= list(subregion_distances.keys()) # get the labels of the subregions
        subregion_values= list(subregion_distances.values()) # get the distance value of the cluster COM from of the subregion COM
        
        min_distance = min(subregion_values) # get the minimum distance
        
        min_index = subregion_values.index(min_distance) # get the index of the minimum distance (= subregion label)
        
        cluster_region = subregion_keys[min_index]
        
        cluster_in_subregion[cluster_key] =  cluster_region
        
    return cluster_in_subregion