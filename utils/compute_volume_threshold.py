def compute_volume_threshold(difference_maps_int_all, percentile_values):
    
    '''
    This function computes the volume threshold for the T2-Cluster analysis.
    
    Parameters:
    
    difference_maps_all (array): A 4D NumPy array containing the intensity thresholded 3D difference maps (axes 0-2) for all subjects (axis 3).
    percentile_values = A list of percentile values to calculate the volume threshold at
    
    '''
    
    import os
    import numpy as np
    import seaborn as sns
    import nibabel as nib
    import pandas as pd
    import matplotlib.pyplot as plt
    from utils.erode import erode
    import cc3d
    from utils.append_df_to_excel import append_df_to_excel
    from utils.connected_component_analysis import connected_component_analysis


    # Initialize an empty list to store the non-NaN values from all images
    num_voxel_all = np.array([])
    
    # Convert image to binary
    map_binary= np.where(~np.isnan(difference_maps_int_all), 1, 0)
    
    # count the number of ones
    num_ones= np.count_nonzero(map_binary)
    
    for sub in range(difference_maps_int_all.shape[3]):

        labels, total_labels= cc3d.connected_components(map_binary.astype(np.uint64)[:,:,:,sub], connectivity=6, return_N=True)
        all_voxels= cc3d.statistics(labels)['voxel_counts'] # Total number of voxels
        
        if total_labels > 0:
            all_voxels.sort() # sort in ascending order
            # remove last element which is the background label
            all_voxels_sub= all_voxels[:-1]
        
            # append into a larger array
            num_voxel_all= np.append(num_voxel_all, all_voxels_sub)
    
    # sort them in order
    num_voxel_ordered = np.sort(num_voxel_all) 
    
    # Initialize an empty list to collect data
    volume_threshold_list = []

    # Plotting the histogram
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.histplot(num_voxel_ordered,  
                bins='auto', 
                color='#B8860B', 
                stat='density',
                ax=ax
                )

    ax.set_xlim(0, 50)

    # Calculate percentile values and collect them in a list of dictionaries
    for percentile_value in percentile_values:
        vol_threshold_value = np.percentile(num_voxel_ordered, percentile_value)
        volume_threshold_list.append({'Percentile': percentile_value, 'Volume Threshold': vol_threshold_value})

        # Draw a vertical line at 2 standard deviations above the mean
        plt.axvline(x=vol_threshold_value, color='#8B3E2F', linestyle='dashed', linewidth=1.5)
        
        # Annotate the lines with standard deviation values
        plt.text(vol_threshold_value, plt.ylim()[1] * 0.3, f'{percentile_value}th= \n \n  {vol_threshold_value}', fontsize=8, color='#000000')

    plt.xlabel(f'T2 cluster size (voxels)', fontsize=10)
    plt.ylabel('Probability Density', fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title('T2-Cluster Volume Threshold', fontsize=12)
    plt.grid(True, alpha=0.5, linewidth=0.25)
    plt.show()
    
    # Convert the list of dictionaries to a DataFrame
    volume_threshold_data = pd.DataFrame(volume_threshold_list)
    
    print(volume_threshold_data)

