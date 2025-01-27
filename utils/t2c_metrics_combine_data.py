from os import remove


def t2c_metrics_combine_data(t2c_subregion_data, fc_subregions_path, save_path):
    
    import os
    import numpy as np
    import pandas as pd
    import nibabel as nib

    from utils.append_df_to_excel import append_df_to_excel
    from utils.get_num_voxels import get_num_voxels

    # Load the data from the excel file
    
    fc_subregions = nib.load(fc_subregions_path)
    fc_subregions_mask= fc_subregions.get_fdata().astype(int)
    subregion_num_voxels_dict = get_num_voxels(fc_subregions_mask)
    
    subregions_map = {'AN': 11, 'MC': 12, 'LC': 13, 'MP': 14, 'LP': 15}
    

    # Regions to ensure all are included
    all_regions = ['AN', 'LC', 'LP', 'MC', 'MP']

    # Group by Region and calculate required value    
    
    agg_data = t2c_subregion_data.groupby('Region').agg(
    Total_T2C_Voxels=('T2C Voxels', 'sum'),
    Total_Region_Voxels=('Region Voxels', 'first'),
    T2C_Size=('T2C Size (mm^3)', 'mean'),
    T2C_Count=('T2C Count', 'sum'),
    T2C_Mean=('T2C Mean (ms)', 'mean'), 
    T2C_Std=('T2C Std (ms)', 'mean'),
    T2C_Median=('T2C Median (ms)', 'mean')
    ).reset_index()
    
    data_all = pd.DataFrame()
    
    data_all ['Region'] = agg_data['Region']

    # Calculate other columns
    data_all['T2C Percent'] = (agg_data['Total_T2C_Voxels'] / agg_data['Total_Region_Voxels']) * 100
    data_all['T2C Size (mm^3)'] = agg_data['T2C_Size']
    data_all['T2C Count'] = agg_data['T2C_Count']   
    data_all['T2C Mean (ms)'] = agg_data['T2C_Mean']
    data_all['T2C Std (ms)'] = agg_data['T2C_Std']
    data_all['T2C Median (ms)'] = agg_data['T2C_Median']
    data_all['Region Voxels'] = agg_data['Total_Region_Voxels']
    data_all['T2C Voxels'] = agg_data['Total_T2C_Voxels']

    # Add missing regions with zero values
    missing_regions = set(all_regions) - set(data_all['Region'])
    data_mission= t2c_subregion_data[t2c_subregion_data['Region'].isin(missing_regions)]
    missing_data = pd.DataFrame([{
        'Region': region,
        'T2C Percent': 0,
        'T2C Size (mm^3)': 0,
        'T2C Count': 0,
        'T2C Mean (ms)': 'NaN',
        'T2C Std (ms)': 'NaN',
        'T2C Median (ms)': 'NaN',
        'Region Voxels': subregion_num_voxels_dict[subregions_map[region]],
        'T2C Voxels': 0,
    } for region in missing_regions])

    # Combine the data
    final_data = pd.concat([data_all, missing_data], ignore_index=True)

    # Sort by region to maintain order
    final_data = final_data.sort_values(by='Region').reset_index(drop=True)
    
    sheet = 'T2C Metrics Region-wise'
    append_df_to_excel(final_data, sheet, save_path)
