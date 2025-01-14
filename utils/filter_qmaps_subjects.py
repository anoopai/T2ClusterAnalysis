def filter_qmaps_subjects(subject_data, data_dir_path, dir_path, quant_type, status, seg_mask_type, mask_erode, log_path):
    
    '''
    subject_data= dictionary of the subject data. subject_data['Subject_number']['VISIT-NUMBER']['scan type'] 
    data_dir_path= directory where the subject data is stored, 
    dir_path= Main folder drectory path, 
    quant_type= t1rho or t2,
    '''
    import os
    import numpy as np
    import traceback
    import nibabel as nib

    from utils.filter_qmaps import filter_qmaps

    # %%
    try:

        sub_nums= list(subject_data.keys())

        for sub_num in sub_nums:
            visits= list(subject_data[sub_num].keys())

            for visit in visits:            
                    
                    knees= list(subject_data[sub_num][visit].keys())
                    
                    for knee in knees:

                        scan= subject_data[sub_num][visit][knee]

                        print(scan)
                
                        timepoint = os.path.join(data_dir_path, scan)

                        # walk through the directory
                        for root, dirs, files in os.walk(timepoint):
                             for file in files:         
                                if f'{quant_type}.nii.gz' in file:
                                    qmap_path= os.path.join(root, file)
                                    qmap_save_path = os.path.join(root, f'{file[:-7]}_filtered.nii.gz')

                                    if os.path.exists(qmap_save_path):
                                        print(f'Filtered qmap exists for {scan}')

                                    else:
                                       # Filter the quantitative map
                                       qmap = nib.load(qmap_path)
                                       qmaps_filtered= filter_qmaps(qmap, fwhm= 1)
                                       nib.save(qmaps_filtered, qmap_save_path)
    except Exception as e:
        # raise e
        log_file_path = os.path.join(log_path, 'filter_qmaps_subjects_log.txt')
        with open(log_file_path, 'a') as f:
            f.write(traceback.format_exc())