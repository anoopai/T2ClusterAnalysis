def convert_qdess_dicom2nii(qdess_dicom_load_path, qdess_nii_save_path):

    import os
    import shutil
    import dosma as dm
    import numpy as np
    from dosma.scan_sequences import QDess
    from dosma import ImageDataFormat
    
    '''
    Reads and converts qdess file into nifti
    
    '''
    print("Loading Qdess dicoms...")
    group_by = ("EchoNumbers", "SeriesDescription")
    dr = dm.DicomReader(num_workers=4, group_by=group_by, verbose=True)
    volumes = dr.read(qdess_dicom_load_path)
    volumes = sorted(volumes, key=lambda v: v.get_metadata("EchoNumbers", float))

    # Stuff the volumes into a QDess object.
    qdess = QDess([x.astype(np.float64) for x in volumes])
    if os.path.exists(qdess_nii_save_path):
        shutil.rmtree(qdess_nii_save_path)
        print("Saving Qdess as nifti...")
        qdess.save(qdess_nii_save_path, save_custom=True, image_data_format=ImageDataFormat.nifti)
    else:
        print("Saving Qdess as nifti...")
        qdess.save(qdess_nii_save_path, save_custom=True, image_data_format=ImageDataFormat.nifti)
    

