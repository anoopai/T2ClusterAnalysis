def com_of_labeled_mask(label_mask):
    
    import numpy as np
    from scipy.ndimage import center_of_mass
    
    '''
    Input: label_mask: 3D numpy array with integer values representing different clusters

    Output: com_values: dictionary with cluster labels as keys and center of mass as values    
    '''
    
    com_values= {}
    
    # get unique labels
    labels= np.unique(label_mask)  
    labels= labels[labels > 0]  # remove 0 from labels (background label)
    
    # fine COM for each cluster
    for label in labels:
        component = np.where(label_mask == label, 1, 0) #astype(np.float64)
        # print(component)
        com = center_of_mass(component)
        # print(label, com)
        com_values[label] = tuple(np.round(com, 4))
        
    return com_values