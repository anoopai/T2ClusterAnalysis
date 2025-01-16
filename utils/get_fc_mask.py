def get_fc_mask(seg_path, seg_fc_path):
    
    '''
    
    Get femoral cartilage label from the segmentation
    
    seg_path: str
        Path to the segmentation file
    seg_fc_path: str
        Path to save the femoral cartilage segmentation file
    
    '''

    import os
    import nibabel as nib
    import numpy as np

    if os.path.exists(seg_path):
        seg = nib.load(seg_path)
        fc_seg = np.where(seg.get_fdata() == 2, 1, 0).astype(float)
    #    fc_seg = np.where(np.isin(seg.get_fdata(), [2, 7]), seg.get_fdata(), 0).astype(float) # get femur segmentation
        seg_nii = nib.Nifti1Image(fc_seg, seg.affine)
        print(f'Saving femoral cartilage segmentation to {seg_fc_path}')
        nib.save(seg_nii, seg_fc_path)
    else:
        print(f'{seg_path} does not exist. Please check the path.')