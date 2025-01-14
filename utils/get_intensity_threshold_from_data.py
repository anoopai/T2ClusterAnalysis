def get_intensity_threshold_from_data(subject_data, results_path):
    
    '''
    subject_data= dictionary of the subject data. subject_data['Subject_number']['VISIT-NUMBER']['scan type'] 
    data_dir_path= directory where the subject data is stored, 
    dir_path= Main folder drectory path, 
    quant_type= t1rho or t2, 
    suppression= Fat and Fluid suppression while computing parameter (T1 or T2)
    '''
    import os
    import numpy as np
    import traceback
    import nibabel as nib
    import os
    import numpy as np
    import nibabel as nib
    import pandas as pd
    import matplotlib.pyplot as plt
    from utils.erode import erode
    import openpyxl
    import shutil
    import re
    from utils.append_df_to_excel import append_df_to_excel


    try:

        # Initialize an empty list to store the non-NaN values from all images
        non_nan_values_all = np.array([])

        sub_nums= list(subject_data.keys())

        for sub_num in sub_nums:
            visits= list(subject_data[sub_num].keys())
            
            for visit in visits:
                    
                if visit == 'VISIT-2':

                    # print(sub_num)
                    scan_0= subject_data[sub_num][visit]['NA']
                    timepoint_0 = os.path.join(data_dir_path, scan_0)
                    
                    difference_maps_path= os.path.join(timepoint_0, f'results/cluster_analysis/{quant_type}_{status}_{seg_mask_type}/difference_maps.nii.gz')
                    
                    # load 3D data rolled
                    difference_maps_mv = nib.load(difference_maps_path)
                    difference_maps = difference_maps_mv.get_fdata()

                    # Initialize an empty list to store the non-NaN values from all images
                    non_nan_values = []
                    
                    # Flatten the 3D array and remove NaN values
                    non_nan_values.extend(difference_maps[~np.isnan(difference_maps)])

                    # Convert the list of non-NaN values to a NumPy array
                    non_nan_values = np.array(non_nan_values)
                    non_nan_values_all = np.append(non_nan_values_all, non_nan_values)

                    mean_value_sub = np.mean(non_nan_values)
                    std_dev_sub = np.std(non_nan_values)
                    
                    threshold_controls_subject = pd.DataFrame({
                        'Subject': [sub_num],
                        'Mean': [mean_value_sub],
                    })
                    
                    threshold_controls_subject_pos = pd.DataFrame({
                        'Subject': [sub_num],
                        'Mean': [mean_value_sub],
                    })
                    
                    threshold_controls_subject_neg = pd.DataFrame({
                        'Subject': [sub_num],
                        'Mean': [mean_value_sub],
                    })
 
                    # Calculate the value at 2 standard deviations
                    values = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
                    for value in values:
                        threshold_controls_subject[f'{value}*Std'] = value * std_dev_sub
                        threshold_controls_subject_pos[f'{value}*Std'] = mean_value_sub + value * std_dev_sub
                        threshold_controls_subject_neg[f'{value}*Std'] = mean_value_sub - value * std_dev_sub

                    data_file_path_output_sub = os.path.join(results_path, f'cluster_analysis/intensity_threshold/intensity_threshold_controls_subjectwise.xlsx')

                    if os.path.exists(os.path.dirname(data_file_path_output_sub)) == False:
                        os.mkdir(os.path.dirname(data_file_path_output_sub))
                    
                    # append_df_to_excel(data_file_path_output=data_file_path_output_sub, data=threshold_controls_subject, sheet=f'{quant_type}_{status}', log_path=log_path)  
                    append_df_to_excel(data_file_path_output=data_file_path_output_sub, data=threshold_controls_subject_pos, sheet=f'{quant_type}_{status}_pos', log_path=log_path)  
                    append_df_to_excel(data_file_path_output=data_file_path_output_sub, data=threshold_controls_subject_neg, sheet=f'{quant_type}_{status}_neg', log_path=log_path)  


            array_save_path= os.path.join(results_path, f'cluster_analysis/intensity_threshold/{quant_type}_{status}')
            np.save(array_save_path, non_nan_values_all)
        
            plot_save_path= os.path.join(results_path, f'cluster_analysis/intensity_threshold/{quant_type}_{status}.png')
            if os.path.exists(plot_save_path) == False:

                # Plot a histogram of the non-NaN values
                plt.figure(figsize=(10, 10))
                
                #automatic bin size
                plt.hist(non_nan_values_all, bins= 'auto', color='blue', alpha=0.7)
                # plt.hist(non_nan_values, bins= 100, color='blue', alpha=0.7)
                # plot from -90 to +90 on x-axis
                plt.xlim(-25, 25)
                # if quant_type== 't2':
                #     plt.ylim(0, 4000)
                # elif quant_type== 't1rho':
                #     plt.ylim(0, 5000)
                
                plt.xlabel(f'{quant_type}', fontsize=15)
                plt.ylabel('Number of Voxels', fontsize=15)
                plt.title('Difference map voxels of Visits 1 & 2 for Controls', fontsize=15)
                plt.grid(True)
                plt.savefig(plot_save_path)
                plt.close()

        # Calculate the mean and standard deviation of the non-NaN values
        mean_value = np.mean(non_nan_values_all)
        std_dev = np.std(non_nan_values_all)
        
        threshold_controls = pd.DataFrame({
            'Mean': [mean_value],
        })
        
        threshold_controls_pos = pd.DataFrame({
            'Mean': [mean_value],
        })
        
        threshold_controls_neg = pd.DataFrame({
            'Mean': [mean_value],
        })

        # Calculate the value at 2 standard deviations
        values = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
        for value in values:
            threshold_controls[f'{value}*Std'] = value * std_dev
            threshold_controls_pos[f'{value}*Std'] = mean_value + value * std_dev
            threshold_controls_neg[f'{value}*Std'] = mean_value - value * std_dev

        data_file_path_output = os.path.join(results_path, f'cluster_analysis/intensity_threshold/intensity_threshold_controls.xlsx')
        
        # append_df_to_excel(data_file_path_output=data_file_path_output, data=threshold_controls, sheet=f'{quant_type}_{status}', log_path=log_path)  
        append_df_to_excel(data_file_path_output=data_file_path_output, data=threshold_controls_pos, sheet=f'{quant_type}_{status}_pos', log_path=log_path)  
        append_df_to_excel(data_file_path_output=data_file_path_output, data=threshold_controls_neg, sheet=f'{quant_type}_{status}_neg', log_path=log_path)  

        # combine into one excell sheet
        xls = pd.ExcelFile(data_file_path_output)

        # Initialize an empty DataFrame to store the combined data
        combined_data = pd.DataFrame()

        # Iterate through sheet names
        for sheet_name in xls.sheet_names:
            
            # Read data from the current sheet ignoring the 1st row (header)
            data = pd.read_excel(data_file_path_output, sheet_name=sheet_name)

            pattern = r'(t2|t1_rho)_(reg2baseline|reg2timepoint)_(pos|neg)'

            match = re.match(pattern, sheet_name)
            quant_type = match.group(1)
            registration = match.group(2)
            cluster_type = match.group(3)
            
            # Add a "Conditions" column with the sheet name
            data.insert(0, 'Quant type', quant_type) 
            data.insert(1, 'Registration', registration)
            data.insert(2, 'Cluster type', cluster_type) 
            
            # Append data to the combined_data DataFrame
            combined_data = pd.concat([combined_data, data], ignore_index=True)

        # with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='w') as writer:
        # Write the DataFrame to a new sheet (e.g., 'NewSheet')
        excel_file_path = os.path.join(results_path, f'cluster_analysis/intensity_threshold/intensity_threshold_controls_combined.xlsx')
            
        combined_data.to_excel(excel_file_path, sheet_name='Intensity_threshold', index=False)

        # delete all sheets except intensity_threshold
        # https://stackoverflow.com/questions/43375010/how-to-delete-a-sheet-in-excel-using-python

        wb = openpyxl.load_workbook(excel_file_path)
        for sheet in wb.worksheets:
            if sheet.title != 'Intensity_threshold':
                wb.remove(sheet)

    except Exception as e:
        # raise e
        log_file_path = os.path.join(log_path, 'aggregate_intensity_thresholds_log.txt')
        with open(log_file_path, 'a') as f:
            f.write(traceback.format_exc())
