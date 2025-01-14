def realign_seg_masks_dosma_format(seg):
    
    ''' 
    realigns the segmentation masks to the format used by DOSMA
    Input: seg = sitk Image
    
    '''
    
    import SimpleITK as sitk
    import numpy as np
    
    # seg = sitk.ReadImage(seg_path)
    array = sitk.GetArrayFromImage(seg)
    array = np.transpose(array, (0, 2, 1)).astype(int)
    seg_ = sitk.GetImageFromArray(array)
    spacing = seg.GetSpacing()
    seg_.SetSpacing((spacing[1], spacing[0], spacing[2]))
    seg_.SetOrigin(seg.GetOrigin())
    
    direction = seg.GetDirection()
    direction = np.array(direction).reshape((3, 3))
    direction_ = np.zeros((3,3))
    direction_[:,0] = direction[:,1]
    direction_[:,1] = direction[:,0]
    direction_[:,2] = direction[:,2]
    seg_.SetDirection(direction_.flatten())
    
    # cast to int
    seg_ = sitk.Cast(seg_, sitk.sitkInt16)
    # sitk.WriteImage(seg_, seg_save_path, useCompression=True)
    
    return seg_
    
    