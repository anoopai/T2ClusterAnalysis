def t2_subregion_means(t2_map_path, fc_subregions_path, save_path, sheet):
    
    '''
    Calculate the mean T2 value for each subregion and save to an excel file
    
    Parameters
    ----------
    t2_map_path : str
        Path to the T2 map file
    fc_subregions_path : str
        Path to the FC subregions file
    save_path : str
        Path to save the excel file   
    '''
    
    import os
    import numpy as np
    import pandas as pd
    import nibabel as nib

    from utils.append_df_to_excel import append_df_to_excel

    # Load the data from the excel file
    t2_map = nib.load(t2_map_path).get_fdata()
    
    fc_subregions = nib.load(fc_subregions_path)
    fc_subregions_mask= fc_subregions.get_fdata().astype(int)
    
    # Calculate the mean T2 value for each subregion
    subregion_labels = np.unique(fc_subregions_mask)
    
    subregions_map = {11:'AN', 12:'MC', 13:'LC', 14:'MP', 15:'LP'}
    
    subregion_df= pd.DataFrame()
    
    for label in subregion_labels:
        if label == 0:
            continue
        mask = (fc_subregions_mask == label)
        mean_t2 = np.mean(t2_map[mask])
        
        subregion_dict = {
            'Region': subregions_map[label],
            'T2 Mean': [mean_t2],
            'T2 Std': [np.std(t2_map[mask])],
            'T2 Median': [np.median(t2_map[mask])]}
        
        subregion_df = pd.concat([subregion_df, pd.DataFrame(subregion_dict)], ignore_index=True)
        
    append_df_to_excel(subregion_df, sheet, save_path)
