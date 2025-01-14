def erode(input_file, size):

    from skimage.morphology import square
    import cv2
    import numpy as np

    # make all the nan in the image to 0
    input_file_no_nan= np.nan_to_num(input_file, copy=True, nan=0.0)

    # Convert image to binary
    input_file_binary = np.where(input_file_no_nan != 0, 1, 0)
    input_file_binary = input_file_binary.astype('uint8')

    # Define a square structural element to erode the image
    # kernel= square(3)

    # Elliptical Kernel
    kernel= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size))

    input_file_binary_eroded = cv2.erode(input_file_binary, kernel=kernel)
    
    # get the original values of the image back after erosion
    input_file_processed= input_file * input_file_binary_eroded
    
    # input_file_binary_dilated = cv2.dilate(input_file_binary, kernel=kernel)

    return input_file_processed