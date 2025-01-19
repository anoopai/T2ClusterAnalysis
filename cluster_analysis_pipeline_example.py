import os
import sys
import dosma as dm
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import pickle
import json
from pathlib import Path
from dosma.scan_sequences import QDess
from dosma import ImageDataFormat

from utils.dosma_segmentation_bone_cartilage import *
from utils.get_fc_mask import *
from utils.get_fc_and_femur_masks import * 
from utils.visualize_sgd_registration import *
from utils.compute_t2_fc import *
from utils.filter_t2maps import *
from utils.visualize_t2_maps import *
from utils.visualize_t2_maps_fc import *
from utils.t2_difference_maps import *
from utils.apply_intensity_threshold import *
from utils.apply_size_threshold import *
from utils.visualize_t2_cluster_analysis import *
from utils.convert_qdess_dicom2nii import *
from utils.sgd_based_registration import *
from utils.sgd_based_qmap_registration import *
from utils.dosma_segmentation import *
from utils.compute_DSC import *
from utils.compute_intensity_threshold import *
from utils.compute_size_threshold import * 

######################################################################################################################

# Read the config file
path_config = '/dataNAS/people/anoopai/T2ClusterAnalysis/config.json'

with open(path_config) as f:
    config = json.load(f)
    
image1_dicom_path = config['image1_dicom_path']
image2_dicom_path = config['image2_dicom_path']

image1_path =  config['image1_path']
image2_path =  config['image2_path']

results_path = config['results_path']
weights_path = config['dosma_weights_path']
elastix_reg_parameters_path = config['elastix_registration_parameters_path']

cluster_type = config['cluster_type']
intensity_threshold = config['intensity_threshold']
size_threshold = config['size_threshold']

#####################################################################################################################
#####################################################################################################################
# I. Data Preprocessing 
#####################################################################################################################
#####################################################################################################################

# 1. Convert Qdess file format: dicoms to nii (usable by Dosma)
if image1_dicom_path != "":
    # image1 scans
    convert_qdess_dicom2nii(image1_dicom_path, image1_path)

if image2_dicom_path != "":
    # image2 scans
    convert_qdess_dicom2nii(image2_dicom_path, image2_path)
    
#####################################################################################################################

# 2. Segment the cartilage and bone
# ATTENTION: Needs GPU!
seg1_path= os.path.join(results_path, 'seg1_all.nii.gz')
# print("Segmenting image 1")
# dosma_segmentation_bone_cartilage(
#         qdess_file_path = image1_path,
#         output_file_path = seg1_path,
#         weights_path = weights_path)

seg2_path= os.path.join(results_path, 'seg2_all.nii.gz')
# print("Segmenting image 2")
# dosma_segmentation_bone_cartilage(
#         qdess_file_path = image2_path,
#         output_file_path = seg2_path,
#         weights_path = weights_path)

# Save femoral cartilage and femur masks Seperately

seg1_fc_path = os.path.join(results_path, 'seg1_fc.nii.gz')
seg2_fc_path = os.path.join(results_path, 'seg2_fc.nii.gz')

# get_fc_mask(seg1_path, seg1_fc_path)
# get_fc_mask(seg2_path, seg2_fc_path)

seg1_fc_femur_path = os.path.join(results_path, 'seg1_fc_and_femur.nii.gz')
seg2_fc_femur_path = os.path.join(results_path, 'seg2_fc_and_femur.nii.gz')

# get_fc_and_femur_masks(seg1_fc_path, seg1_fc_femur_path)
# get_fc_and_femur_masks(seg2_fc_path, seg2_fc_femur_path)

#####################################################################################################################

# II. Signed distance field-based Registration of Images from different Visits of the same subject 
# followed by re-segmentation. Perform checks (i) visually and (ii) computing Dice Similarity Coefficient (DSC)

#####################################################################################################################

# 1. Registration
image2_reg_path = os.path.join(results_path, 'image2_reg')

# sgd_based_registration(
#     fixed_img_path = image1_path,
#     moving_img_path = image2_path,
#     moving_img_save_path = image2_reg_path,
#     fixed_mask_path = seg1_fc_path,
#     moving_mask_path = seg2_fc_path,
#     elastix_file_path = elastix_reg_parameters_path,
#     reg_path = results_path
#    ) 

###################################################################################################################

# 2. Re-segmentation
# Option 1
seg2_reg_path= os.path.join(results_path, 'seg2_reg.nii.gz')
# print("Resegmenting image 2 after registration to image 1")
# dosma_segmentation_bone_cartilage(
#         qdess_file_path = image2_reg_path,
#         output_file_path = seg2_reg_path,
#         weights_path = weights_path)

# Option 2
# image2_mask_reg_path = image1_mask_path

seg2_fc_reg_path = os.path.join(results_path, 'seg2_fc_reg.nii.gz')
# get_fc_mask(seg2_reg_path, seg2_fc_reg_path)

seg2_fc_femur_reg_path = os.path.join(results_path, 'seg2_fc_and_femur_reg.nii.gz')
# get_fc_and_femur_masks(seg2_reg_path, seg2_fc_femur_path)

#####################################################################################################################

# 3. compute Dice Similarity Coefficient of the fc masks from image 1 and resegmented fc mask on image 2 
# after registration of image 2 to image 1

# load masks
# seg1_fc = nib.load(seg1_fc_path).get_fdata()
# seg2_fc_reg = nib.load(seg2_fc_reg_path).get_fdata()
 
# DSC= compute_DSC(
#     mask1= seg1_fc,
#     mask2= seg2_fc_reg
# )

# # Visualise the fc masks from image 1 and resegmented fc mask on image 2 after registration of image 2 to image 1
# visualize_sgd_registration(
#     fixed_img_path = image1_path,
#     moving_img_path = image2_path,
#     moving_img_reg_path = image2_reg_path,
#     fixed_mask_path = seg1_fc_path,
#     moving_mask_path = seg2_fc_path,
#     moving_mask_reg_path = seg2_fc_reg_path,
#     result_path = results_path
# )

# if DSC < 0.75:
#     print(f'Dice score is: {np.round(DSC, 4)}. Registration between Image 1 and Image 2 is not good. Please check the registration')
#     sys.exit('Stopping the run due to poor registration.')
# else:
#     print(f'Dice score is: {np.round(DSC, 4)}. Registration is good')
    
######################################################################################################################    

#  III Quantitative Maps and Preprocessing

######################################################################################################################

# 4. Compute quantitative maps (T2/T1rho) of the tissue (Femoral cartilage in this case)
    
print("Computing T2 maps")

t2_map1_path = os.path.join(results_path, 't2_map1.nii.gz')
t2_map2_path = os.path.join(results_path, 't2_map2.nii.gz')

# # Image1 
# compute_t2_fc(qdess_path = image1_path,
#            mask_path = seg1_fc_path,
#            t2_save_path = t2_map1_path
#            )

# # Image2
# compute_t2_fc(qdess_path = image2_reg_path,
#            mask_path = seg2_fc_reg_path,
#            t2_save_path = t2_map2_path
#            )

######################################################################################################################
# TO DO visualize T2 maps and filtering
# TO DO: Save T2 map region-wise
######################################################################################################################

######################################################################################################################

# 5. Filter T2 maps with a Gaussian filter

t2_map1_filt_path= os.path.join(results_path, 't2_map1_filt.nii.gz')
t2_map2_filt_path= os.path.join(results_path, 't2_map2_filt.nii.gz')

print("Filtering T2 maps")

# # Filter T2 maps
# t2_map1_filt= filter_t2maps(
#     t2_map_path= t2_map1_path, 
#     fwhm= 1, # Check the fwhm value for your used-case!!!!
#     t2_map_save_path= t2_map1_filt_path 
#     ) 

# t2_map2_filt= filter_t2maps(
#     t2_map_path= t2_map2_path, 
#     fwhm= 1, # Check the fwhm value for your used-case!!!!
#     t2_map_save_path= t2_map2_filt_path
#    )

# 6. Save T2 maps for FC-only seperately

t2_map1_fc_path = os.path.join(results_path, 't2_map1_fc_filt.nii.gz')
t2_map2_fc_path = os.path.join(results_path, 't2_map2_fc_filt.nii.gz')

# Image 1
# t2_map1_fc_filt = nib.load(t2_map1_filt_path).get_fdata() * nib.load(seg1_fc_path).get_fdata()
# t2_map1_fc_filt = nib.Nifti1Image(t2_map1_fc_filt, nib.load(t2_map1_filt_path).affine)
# nib.save(t2_map1_fc_filt, t2_map1_fc_path)

# # Image 2
# t2_map2_fc_filt = nib.load(t2_map2_filt_path).get_fdata() * nib.load(seg2_fc_reg_path).get_fdata()
# t2_map2_fc_filt = nib.Nifti1Image(t2_map2_fc_filt, nib.load(t2_map1_filt_path).affine)
# nib.save(t2_map2_fc_filt, t2_map2_fc_path)

# 7. Visualise the T2 maps

print("Visualising and saving T2 maps before and after filtering")
# visualize_t2_maps_fc(
#     t2_map1_path = t2_map1_path, 
#     t2_map1_filt_path = t2_map1_filt_path, 
#     t2_map2_path= t2_map2_path, 
#     t2_map2_filt_path= t2_map2_filt_path, 
#     result_path = results_path
#     )

########################################################################################################################

# III. T2 Cluster Analysis

########################################################################################################################

# 1. Compute difference map of T2 values between the two time points (Image 2 minus Image 1)

t2_diff_map_path = os.path.join(results_path, 't2_difference_map.nii.gz')

print("Computing difference map of T2 values between the two time points (Image 2 minus Image 1)")
t2_difference_map = t2_difference_maps(
    baseline_qmap_path = t2_map1_fc_path, 
    followup_qmap_path = t2_map2_fc_path, 
    baseline_mask_path = seg1_fc_path, 
    followup_mask_path = seg2_fc_reg_path, 
    mask_erode=False, 
    erode_size=1,
    diff_map_save_path = t2_diff_map_path
    )


# 2. Apply intensity threshold to the difference map

t2_intensity_threshold_path = os.path.join(results_path, 't2_difference_map_int_threshold.nii.gz')

print(f"Applying intensity threshold [{intensity_threshold} (ms)] to the difference map")
diff_map_intensity_thresholded = apply_intensity_threshold(
    difference_map_path= t2_diff_map_path, 
    intensity_threshold= intensity_threshold,  # change this as per your data
    cluster_type= cluster_type, 
    save_path= t2_intensity_threshold_path
    )

# 3. Apply size threshold to the difference map

t2_size_threshold_path = os.path.join(results_path, 't2_difference_map_size_threshold.nii.gz')

print(f"Applying size threshold [{size_threshold} (voxels)] to the difference map")
diff_map_size_thresholded = apply_size_threshold(
    difference_map_path= t2_intensity_threshold_path, 
    size_threshold= size_threshold, # change this as per your data
    save_path= t2_size_threshold_path
    ) 

# 4. Visualise T2 clusters
print("Visualising T2 clusters on a sagittal slice ")

visualize_t2_cluster_analysis(
    t2_difference_map_path= t2_diff_map_path, 
    t2_int_threshold_path= t2_intensity_threshold_path, 
    t2_size_threshold_path= t2_size_threshold_path, 
    result_path = results_path
    )

