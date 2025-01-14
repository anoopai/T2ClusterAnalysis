def gaussian_filter(map, clip_value):

# gaussian filter with FWHM= 2 mm

    from scipy.ndimage import gaussian_filter
    import numpy as np

    # gaussian filter with FWHM= 2 mm
    map_filtered = gaussian_filter(map, sigma= 0.5)
    