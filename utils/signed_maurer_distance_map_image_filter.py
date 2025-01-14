def signed_maurer_distance_map_image_filter(input_array):
    
    import SimpleITK as sitk
    import numpy as np

    """ Signed Maurer Distance using SimpleITK

    Args:
        input array: This should be a mask with values of 0 and 1.

    Returns:
        numpy array of signed distances of the input
    """
    input_array= input_array.astype(np.int64)
    signed_dist_filter = sitk.SignedMaurerDistanceMapImageFilter()
    signed_dist_filter.SetSquaredDistance(False)
    signed_dist_filter.SetUseImageSpacing(True)
    mask_image = sitk.GetImageFromArray(input_array)
    signed_image = signed_dist_filter.Execute(mask_image)
    signed_array = sitk.GetArrayFromImage(signed_image)
    return signed_array