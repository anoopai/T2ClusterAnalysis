def filter_qmaps(qmap_path, fwhm):

    '''
    Filter the quantitative maps with a Gaussian filter.
    :param qmap: quantitative map
    :param fwhm: full width at half maximum (FWHM) in mm
    :return: filtered quantitative map

    '''

    import nibabel as nib
    import numpy as np
    from scipy.ndimage import gaussian_filter

    # Get the image data as a numpy array
    qmap= nib.load(qmap_path)
    data = qmap.get_fdata()

    # Define the voxel sizes in mm
    voxel_sizes = qmap.header.get_zooms()

    # Calculate the sigma value for the Gaussian filter
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))) / voxel_sizes

    # Apply the Gaussian filter to the image data
    filtered_data = gaussian_filter(data, sigma)

    # Create a new NIfTI image with the filtered data
    filtered_img = nib.Nifti1Image(filtered_data, qmap.affine)

    return filtered_img