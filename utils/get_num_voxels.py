def get_num_voxels(label_mask):
    
    '''
    Input: label_mask: 3D numpy array with integer values representing different clusters

    Output: Num_voxels: dictionary with cluster labels as keys and number of voxels per label as values    
    '''
    
    import numpy as np
    
    num_voxels= {}

    labels = np.unique(label_mask)
    labels = labels[labels > 0]
    for label in labels:
        mask = (label_mask == label)
        count_voxels = np.sum(mask)
        num_voxels[label] = count_voxels
        
    return num_voxels