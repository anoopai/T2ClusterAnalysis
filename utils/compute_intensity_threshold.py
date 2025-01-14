def compute_intensity_threshold(difference_maps_all, std_values):
    
    '''
    This function computes the intensity threshold for the T2-Cluster analysis.
    
    Parameters:
    
    difference_maps_all (array): A 4D NumPy array containing the 3D difference maps (axes 0-2) for all subjects (axis 3).
    std_vales = A list of standard deviation values to calculate the intensity threshold at
    
    '''
    

    import os
    import numpy as np
    import nibabel as nib
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from utils.append_df_to_excel import append_df_to_excel

    # Initialize an empty list to store the non-NaN values from all images
    non_nan_values_all = np.array([])
    
    # Initialize an empty list to store the non-NaN values from all images
    non_nan_values = []
    
    # Flatten the 3D array and remove NaN values
    non_nan_values.extend(difference_maps_all[~np.isnan(difference_maps_all)])

    # Convert the list of non-NaN values to a NumPy array
    non_nan_values_all = np.array(non_nan_values)

    mean_value = np.mean(non_nan_values)
    std_dev = np.std(non_nan_values)
    
    intensity_threshold_data = pd.DataFrame({
            'Mean': [mean_value],
        })

    # Calculate the value at x standard deviations
    for std_value in std_values:
        intensity_threshold = mean_value + std_value * std_dev
        intensity_threshold_data[f'{std_value}*Std'] = mean_value + std_value * std_dev
        
    print(intensity_threshold_data)
        
    fig, ax= plt.subplots(figsize=(5, 5))
    sns.histplot(non_nan_values_all,  
                # kde=True,
                bins='auto', 
                color = '#B8860B', 
                # hist_kws={'edgecolor':'#B8860B'},
                # kde_kws={'linewidth': 1},
                stat = 'density',
                ax=ax
                )
    ax.set_xlim(-20, 20)
    
    for std_value in std_values:

        # Calculate the value at 2 standard deviations
        stdev = mean_value + std_value * std_dev

        # Draw a vertical line at 2 standard deviations above the mean
        plt.axvline(x=stdev, color='#8B3E2F', linestyle='dashed', linewidth=1.5)
        
        # Annotate the lines with standard deviation values
        plt.text(stdev, plt.ylim()[1] * 0.8, f'{std_value}SD= \n \n  {stdev:.2f}', fontsize=10, color='#000000')

    plt.xlabel(f'T2 change (ms)', fontsize=10)
    plt.ylabel('Probability Density', fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title('T2-Cluster Intensity Threshold', fontsize=12)
    plt.grid(True, alpha=0.5, linewidth=0.25)
    plt.show()
