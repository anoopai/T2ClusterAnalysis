def eucledian_distance_t2c_subregions(data1_dict, data2_dict):
    
    import numpy as np
    
    '''
    input:
    data1_dict = dictionary of fc subregions with keys as labels and values as COM
    data2_dict = dictionary of t2c with keys as labels and values as COM
     
    output:
    
    dict: nested dictionary with 
        key1: (label X from data1_dict), 
        Key1_1= label Y from data2_dict),
        value: eucledian distance between the two COMs
     
    '''
    cluster_subregion_distances = {}
  
    for key1, point1 in data2_dict.items():
        
        fc_region = {}
        
        # Convert point1 to a NumPy array if it's a tuple
        if isinstance(point1, tuple):
            point1 = np.array(point1)
        
        for key2, point2 in data1_dict.items():
            # Convert point2 to a NumPy array if it's a tuple
            if isinstance(point2, tuple):
                point2 = np.array(point2)
            
            distance = np.linalg.norm(point1 - point2)
            fc_region[key2] = distance
            cluster_subregion_distances[key1] = fc_region
    
    return cluster_subregion_distances