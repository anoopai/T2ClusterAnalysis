import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import pickle
import pandas as pd
import SimpleITK as sitk
import dosma as dm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from pathlib import Path
from dosma.scan_sequences import QDess
from dosma import ImageDataFormat

from utils.convert_qdess_dicom2nii import *
from utils.dosma_segmentation_bone_cartilage import *
from utils.append_df_to_excel import *
from utils.filter_qmaps import *
from T2ClusterAnalysis.utils.compute_difference_map import *
from utils.apply_intensity_threshold import *
from utils.apply_volume_threshold import *
from utils.compute_t2_DODGR import *

from utils.sgd_based_registration import *
from utils.sgd_based_qmap_registration import *
from utils.dosma_segmentation import *
from utils.compute_DSC import *
from utils.compute_intensity_threshold import *
from utils.compute_volume_threshold import * 
from utils.compute_T2C_metrics import * 
from utils.append_df_to_excel import *  
# from utils.generate_fc_subregions import * # pymskt environment
from utils.assign_t2c_to_subregions import *


dir_path= Path('/dataNAS/people/anoopai/DESS_ACL_study/')
data_path= dir_path / 'data/'
files_path= dir_path / '/dataNAS/people/anoopai/T2_Cluster_Analysis/files/'
reg_check_path= dir_path / '/dataNAS/people/anoopai/T2_Cluster_Analysis/files/registration_check'
seg_check_path= dir_path / '/dataNAS/people/anoopai/T2_Cluster_Analysis/files/segmentation_check'
results_path= dir_path / 'results/27Nov2024'

# # Set elastix environment variables
# elastix_folder = '/dataNAS/people/anoopai/elastix'
# os.environ['PATH'] = f"{elastix_folder}/bin:{os.environ['PATH']}"
# if 'LD_LIBRARY_PATH' in os.environ:
#     os.environ['LD_LIBRARY_PATH'] = f"{elastix_folder}/lib:{os.environ['LD_LIBRARY_PATH']}"
# else:
#     os.environ['LD_LIBRARY_PATH'] = f"{elastix_folder}/lib"

# #!elastix --version
try:
    version_output = subprocess.check_output(['elastix', '--version'], stderr=subprocess.STDOUT, text=True)
    print("Elastix version information:")
    print(version_output)
except subprocess.CalledProcessError as e:
    print("Error running elastix --version:")
    print(e.output)

##################################################################################################################################
# Step 1: Convert DICOM to NIFTI
# conda activate dess
##################################################################################################################################
# if os.path.exists(dicom_data_path):
#     for sub_dir in os.listdir(dicom_data_path):
#         sub_dir_path = os.path.join(dicom_data_path, sub_dir)
#         sub_dir_path2 = os.path.join(data_path, sub_dir)
#         if not os.path.exists(sub_dir_path2):
#             os.makedirs(sub_dir_path2)
#         for data_dir in os.listdir(sub_dir_path):
#             data_dir_path = os.path.join(sub_dir_path, data_dir)
#             data_dir_path2 = os.path.join(sub_dir_path2, f'{data_dir}/qdess')
#             if not os.path.exists(data_dir_path2):
#                 os.makedirs(data_dir_path2)
#             if not os.path.exists(data_dir_path2):
#                 convert_qdess_dicom2nii(data_dir_path, data_dir_path2)
#             else:
#                 print(f'{data_dir_path2} already exists')
            
##############################################################################################################################################
# Step 2: Perform Automatic segmentation: Femoral Cartilage, Medial and Lateral Tibial Cartilage, Patellar Cartilage, Meniscus, Femur, Tibia, Patella
# Note: Requires GPU
# conda activate dosma2
##############################################################################################################################################

weights_path = os.path.join(files_path, 'dosma_weights/Goyal_Bone_Cart_July_2024_best_model.h5')
cmap = mcolors.ListedColormap([    
'#fabebe',  # Pink
'#f9d607',  # Yellow
'#e6194b',  # Red
'#4c11c3',  # Purple
'#46f0f0',  # Cyan
'#f58231',  # Orange
'#f032e6',  # Magenta
'#0da00a',  # Green
'#dbef60',  # Lime
'#4363d8',  # Blue
])

for sub_dir in os.listdir(data_path):
    sub_dir_path = os.path.join(data_path, sub_dir)
    for visit_dir in os.listdir(sub_dir_path):
        visit_dir_path = os.path.join(sub_dir_path, visit_dir)
        for knee_dir in os.listdir(visit_dir_path):
            knee_dir_path = os.path.join(visit_dir_path, knee_dir)
            qdess_path = os.path.join(knee_dir_path, 'scans/qdess')
            segmentation_dir = os.path.join(knee_dir_path, 'segmentation')
            if not os.path.exists(segmentation_dir):
                os.makedirs(segmentation_dir)
            seg_path = os.path.join(knee_dir_path, 'segmentation/segmentation.nii')
            if not os.path.exists(seg_path):
                print(f'Segmenting {qdess_path}')
                dosma_segmentation_bone_cartilage(
                        qdess_file_path = qdess_path,
                        output_file_path = seg_path,
                        weights_path = weights_path)
                if not os.path.exists(seg_path):
                    os.mkdir(seg_path)
                # Check segmentation
                seg_img_save_path= Path(seg_check_path) / f'{sub_dir}_{visit_dir}_{knee_dir}.jpg'
                qdess = QDess.load(qdess_path)
                seg = nib.load(seg_path).get_fdata()
                seg = np.where(seg==0, np.nan, seg)
                slices= [20, 30, 40, 50, 60]
                fig, ax = plt.subplots(1, 5, figsize=(20, 5))
                for i, n in enumerate(slices):
                    ax[i].imshow(qdess.volumes[0].A[:, :, n], cmap='gray')
                    ax[i].imshow(seg[:, :, n], alpha=0.5, cmap=cmap, vmin=0, vmax=9)
                    ax[i].set_xticks([])
                    ax[i].set_yticks([])
                # Adjust layout to make room for the title
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.suptitle(f'{sub_dir} {visit_dir} {knee_dir}', fontsize= 20)
                plt.savefig(seg_img_save_path, bbox_inches='tight')
                plt.close()
            # else:
            #     print(f'{seg_path} already exists')

##############################################################################################################################################
# Step 2B: Check segmentation- Visual check 
##############################################################################################################################################
# for sub_dir in os.listdir(data_path):
#     sub_dir_path = os.path.join(data_path, sub_dir)
#     for visit_dir in os.listdir(sub_dir_path):
#         visit_dir_path = os.path.join(sub_dir_path, visit_dir)
#         for knee_dir in os.listdir(visit_dir_path):
#             knee_dir_path = os.path.join(visit_dir_path, knee_dir)
#             qdess_path = os.path.join(knee_dir_path, 'scans/qdess')
#             segmentation_dir = os.path.join(knee_dir_path, 'segmentation')
#             seg_path = os.path.join(knee_dir_path, 'segmentation/segmentation.nii')
#             seg_img_save_path= Path(seg_check_path) / f'{sub_dir}_{visit_dir}_{knee_dir}.jpg'
#             if os.path.exists(seg_path) and os.path.getsize(qdess_path):
#                 qdess = QDess.load(qdess_path)
#                 seg = nib.load(seg_path).get_fdata()
#                 seg = np.where(seg==0, np.nan, seg)
#                 slices= [20, 30, 44, 50, 60]
#                 fig, ax = plt.subplots(1, 5, figsize=(20, 5))
#                 for i, n in enumerate(slices):
#                     ax[i].imshow(qdess.volumes[0].A[:, :, n], cmap='gray')
#                     ax[i].imshow(seg[:, :, n], alpha=0.5, cmap='Set1')
#                     ax[i].set_xticks([])
#                     ax[i].set_yticks([])
#                 # Adjust layout to make room for the title
#                 fig.tight_layout(rect=[0, 0.03, 1, 0.95])
#                 plt.suptitle(f'{sub_dir} {visit_dir} {knee_dir}', fontsize= 20)
#                 plt.savefig(seg_img_save_path, bbox_inches='tight')
#                 plt.close()
                    
##############################################################################################################################################
# Step 3: Save only FC segmentation as a seperate file for easy accessibility in the future
# conda activate dess
##############################################################################################################################################
           
# for sub_dir in os.listdir(data_path):
#     sub_dir_path = os.path.join(data_path, sub_dir)
#     for visit_dir in os.listdir(sub_dir_path):
#         visit_dir_path = os.path.join(sub_dir_path, visit_dir)
#         for knee_dir in os.listdir(visit_dir_path):
#             knee_dir_path = os.path.join(visit_dir_path, knee_dir)
#             qdess_path = os.path.join(knee_dir_path, 'scans/qdess')
#             segmentation_dir = os.path.join(knee_dir_path, 'segmentation')
#             seg_path = os.path.join(knee_dir_path, 'segmentation/segmentation.nii')
            # seg_save_path = os.path.join(knee_dir_path, 'segmentation/segmentation_fc.nii')
            # if os.path.exists(seg_path) and not os.path.exists(seg_save_path):
            #     seg = nib.load(seg_path)
            #     fc_seg = np.where(seg.get_fdata() == 2, 1, 0).astype(float)
            # #    fc_seg = np.where(np.isin(seg.get_fdata(), [2, 7]), seg.get_fdata(), 0).astype(float) # get femur segmentation
            #     seg_nii = nib.Nifti1Image(fc_seg, seg.affine)
            #     print(f'Saving {seg_save_path}')
            #     nib.save(seg_nii, seg_save_path)
            # else:
            #     print(f'{seg_save_path} already exists')
            
##############################################################################################################################################
# Step 4: SGD-based registration of follow-up scans to baseline scans
# Conda activate dess
##############################################################################################################################################

# elastix_file_path = os.path.join(files_path, 'elastic_parameters/elastix_registration_parameters_SDF_mask.txt')
## idensify baseline and follow-up scans for ACLR and contralateral legs
# for sub_dir in os.listdir(data_path):
#     sub_dir_path = os.path.join(data_path, sub_dir)
#     print(sub_dir)
    
#     idx_dirs= [dir for dir in os.listdir(sub_dir_path) if 'IDX' in dir]
#     cl_dirs= [dir for dir in os.listdir(sub_dir_path) if 'CL' in dir]
    
#     knee_dirs = [idx_dirs, cl_dirs]
    
#     for knee_dir in knee_dirs:
#         for session_dir in knee_dir:
#             if 'bsln' in session_dir:
#                 bsln_dir_path = os.path.join(sub_dir_path, session_dir)
#             elif '6mo' in session_dir:
#                 fup_dir_path = os.path.join(sub_dir_path, session_dir)       
         
#         bsln_qdess_path = os.path.join(bsln_dir_path, 'qdess')
#         bsln_seg_path = os.path.join(bsln_dir_path, 'segmentation_fc.nii')
        
#         fup_qdess_path = os.path.join(fup_dir_path, 'qdess')
#         fup_qdess_reg_path = os.path.join(fup_dir_path, 'qdess_reg')
#         fup_seg_path = os.path.join(fup_dir_path, 'segmentation_fc.nii')
        
#         knee= session_dir.split('_')[1]
#         sub = session_dir.split('_')[0]
        
#         if not os.path.exists(fup_qdess_reg_path):
        
#             print(f'Registering 6mo to bsln for {knee} leg of {sub}')
#             sgd_based_registration(
#             fixed_img_path = bsln_qdess_path,
#             moving_img_path = fup_qdess_path,
#             moving_img_save_path = fup_qdess_reg_path,
#             fixed_mask_path = bsln_seg_path,
#             moving_mask_path = fup_seg_path,
#             elastix_file_path = elastix_file_path,
#             reg_path = reg_path,  
#             reg_check = True  # if True, it will save the a jpg pciture of a random slice (n) of the segmentation mask of the fixed image overlayed on the registered moving image)
#             ) 

##############################################################################################################################################
# Step 5: Re-segmentation: Perform Automatic re-segmentation: Femoral Cartilage, Medial and Lateral Tibial Cartilage, Patellar Cartilage, Meniscus, Femur, Tibia, Patella
# Note: Requires GPU
# conda activate dosma2
##############################################################################################################################################

# print ('Re-segmenting after registration')  
# weights_path = os.path.join(files_path, 'weights/best_model.h5')

# for sub_dir in os.listdir(data_path):
#     sub_dir_path = os.path.join(data_path, sub_dir)
#     print(sub_dir_path)
#     for session_dir in os.listdir(sub_dir_path):
#         if '6mo' in session_dir:
#             session_path = os.path.join(sub_dir_path, session_dir)
#             qdess_path = os.path.join(session_path, 'qdess_reg')
#             seg_path = os.path.join(session_path, 'segmentation_reg.nii')
#             if not os.path.exists(seg_path):
#                 print(f'Segmenting {qdess_path}')
#                 dosma_segmentation_bone_cartilage(
#                         qdess_file_path = qdess_path,
#                         output_file_path = seg_path,
#                         weights_path = weights_path)
#             else:
#                 print(f'{seg_path} already exists')
                
##############################################################################################################################################
# Step 6: Save only FC segmentation as a seperate file for easy accessibility in the future
# conda activate dess
##############################################################################################################################################
        
# print ('Saving Fc segmentation masks')   
# for sub_dir in os.listdir(data_path):
#     sub_dir_path = os.path.join(data_path, sub_dir)
#     for session_dir in os.listdir(sub_dir_path):
#         if '6mo' in session_dir:
#             session_path = os.path.join(sub_dir_path, session_dir)
#             seg_path = os.path.join(session_path, 'segmentation_reg.nii')
#             seg_save_path = os.path.join(session_path, 'segmentation_reg_fc.nii')
#             if os.path.exists(seg_path) and not os.path.exists(seg_save_path):
#                 seg = nib.load(seg_path)
#                 fc_seg = np.where(seg.get_fdata() == 2, 1, 0).astype(float)
#                 seg_nii = nib.Nifti1Image(fc_seg, seg.affine)
#                 print(f'Saving {seg_save_path}')
#                 nib.save(seg_nii, seg_save_path)
#             else:
#                 print(f'{seg_save_path} already exists')
                
###############################################################################################################################################
## Step 7: Compute Dice score between pre-registration baseline fc segmentation mask and [post-registration follow-up fc segmentation mask
## conda activate dess
###############################################################################################################################################
# print('Calculating DSC of FC mask pre and post registration')

# dsc_data_path = os.path.join(results_path, 'DSC_data.xlsx')
# dsc_data = pd.DataFrame(columns=['sub', 'Knee', 'DSC'])

# for sub_dir in os.listdir(data_path):
#     sub_dir_path = os.path.join(data_path, sub_dir)
#     print(sub_dir)
    
#     session_dirs= os.listdir(sub_dir_path)
    
#     idx_dirs= [dir for dir in session_dirs if 'IDX' in dir]
#     cl_dirs= [dir for dir in session_dirs if 'CL' in dir]
#     knee_dirs = [idx_dirs, cl_dirs]
    
#     for knee_dir in knee_dirs:
#         for session_dir in knee_dir:
#             if 'bsln' in session_dir:
#                 bsln_dir_path = os.path.join(sub_dir_path, session_dir)
#             elif '6mo' in session_dir:
#                 fup_dir_path = os.path.join(sub_dir_path, session_dir)     
    
#         baseline_seg_path = os.path.join(bsln_dir_path, 'segmentation_fc.nii')
#         followup_seg_path = os.path.join(fup_dir_path, 'segmentation_reg_fc.nii')
        
#         baseline_seg = nib.load(baseline_seg_path).get_fdata()
#         followup_seg = nib.load(followup_seg_path).get_fdata()
        
#         DSC= compute_DSC(
#             mask1= baseline_seg,
#             mask2= followup_seg
#         )
        
#         sub= session_dir.split('_')[0]
#         knee= session_dir.split('_')[1]
#         visit= session_dir.split('_')[2]    
#         dsc_data = pd.concat([dsc_data, pd.DataFrame({'sub': sub, 'Knee': knee, 'DSC': DSC}, index=[0])], ignore_index=True)
    
# if not os.path.exists(dsc_data_path):    
#     append_df_to_excel(data=dsc_data, data_file_path=dsc_data_path, sheet='DSC_data')

###############################################################################################################################################
## Step 8: Compute t2 maps
###############################################################################################################################################
# print (Computing T2 maps)
# for sub_dir in os.listdir(data_path):
#     sub_dir_path = os.path.join(data_path, sub_dir)
    
#     session_dirs= os.listdir(sub_dir_path)
    
#     for session_dir in session_dirs:
        
#         if 'bsln' in session_dir:
#             print(session_dir)
#             session_path = os.path.join(sub_dir_path, session_dir)
#             qdess_path = os.path.join(session_path, 'qdess')
#             mask_path = os.path.join(session_path, 'segmentation_fc.nii')
#             t2_save_path = os.path.join(session_path, 't2')
#             if not os.path.exists(t2_save_path):
#                 compute_t2_DODGR(qdess_path, mask_path, t2_save_path, lateral2medial= True)
#             else:
#                 print(f'{t2_save_path} already exists')

#         elif '6mo' in session_dir:
#             print(session_dir)
#             session_path = os.path.join(sub_dir_path, session_dir)
#             qdess_path = os.path.join(session_path, 'qdess_reg')
#             mask_path = os.path.join(session_path, 'segmentation_reg_fc.nii')
#             t2_save_path = os.path.join(session_path, 't2')
#             if not os.path.exists(t2_save_path):
#                 compute_t2_DODGR(qdess_path, mask_path, t2_save_path, lateral2medial= True)
#             else:
#                 print(f'{t2_save_path} already exists')
                
###############################################################################################################################################
## Step 9: Filter T2 Maps
###############################################################################################################################################
# for sub_dir in os.listdir(data_path):
#     sub_dir_path = os.path.join(data_path, sub_dir)
    
#     session_dirs= os.listdir(sub_dir_path)
    
#     for session_dir in session_dirs:
#         print(session_dir)
#         session_path = os.path.join(sub_dir_path, session_dir)
#         t2_path = os.path.join(session_path, 't2/fc/t2/t2.nii.gz')
#         t2_filt_path = os.path.join(session_path, 't2/fc/t2/t2_filtered.nii.gz')
#         if not os.path.exists(t2_filt_path):
#             print(f'Filtering and saving T2 map.')
#             t2_filt = filter_qmaps(t2_path, fwhm= 1)
#             nib.save(t2_filt, t2_filt_path)
#         else:
#             print(f'{t2_filt_path} already exists')
            
###############################################################################################################################################
## Step 10: Compute Difference Maps
###############################################################################################################################################
    
# print('Computing difference maps')
# for sub_dir in os.listdir(data_path):
#     sub_dir_path = os.path.join(data_path, sub_dir)
    
#     # session_dirs= os.listdir(sub_dir_path)
    
#     idx_dirs= [dir for dir in os.listdir(sub_dir_path) if 'IDX' in dir]
#     cl_dirs= [dir for dir in os.listdir(sub_dir_path) if 'CL' in dir]
    
#     knee_dirs = [idx_dirs, cl_dirs]
    
#     for knee_dir in knee_dirs:
#         for session_dir in knee_dir:
#             if 'bsln' in session_dir:
#                 idx_bsln_dir_path = os.path.join(sub_dir_path, session_dir)
#             elif '6mo' in session_dir:
#                 idx_fup_dir_path = os.path.join(sub_dir_path, session_dir)       
        
#         baseline_t2_path = os.path.join(idx_bsln_dir_path, 't2/fc/t2/t2_filtered.nii.gz')
#         followup_t2_path = os.path.join(idx_fup_dir_path, 't2/fc/t2/t2_filtered.nii.gz')
#         baseline_seg_path = os.path.join(idx_bsln_dir_path, 'segmentation_fc.nii')
#         followup_seg_path = os.path.join(idx_fup_dir_path, 'segmentation_reg_fc.nii')
#         difference_map_save_path = os.path.join(idx_fup_dir_path, 'difference_map.nii')
        
#         if not os.path.exists(difference_map_save_path):
            
#             print(f'Computing and saving difference map for {sub_dir}')
            
#             # Compute and save difference map
#             difference_map = difference_map_tissue(
#                 baseline_qmap_path= baseline_t2_path, 
#                 followup_qmap_path= followup_t2_path,
#                 baseline_mask_path=baseline_seg_path, 
#                 followup_mask_path= followup_seg_path, 
#                 mask_erode= False, 
#                 erode_size= 1)
            
#             nib.save(difference_map, difference_map_save_path)
            
#         else:
#             print(f'Difference map for {sub_dir_path} already exists')
            
###############################################################################################################################################
## Step 11: Apply intensity threshold to difference maps
###############################################################################################################################################
# print('Applying intensity threshold to difference maps')

# intensity_threshold = 8.5 #in ms

# for sub_dir in os.listdir(data_path):
#     sub_dir_path = os.path.join(data_path, sub_dir)
    
#     # session_dirs= os.listdir(sub_dir_path)
    
#     idx_dirs= [dir for dir in os.listdir(sub_dir_path) if 'IDX' in dir]
#     cl_dirs= [dir for dir in os.listdir(sub_dir_path) if 'CL' in dir]
    
#     knee_dirs = [idx_dirs, cl_dirs]
    
#     for knee_dir in knee_dirs:
#         for session_dir in knee_dir:
#             if '6mo' in session_dir:
#                 dir_path = os.path.join(sub_dir_path, session_dir)       
#                 difference_map_path = os.path.join(dir_path, 'difference_map.nii')
#                 difference_map_pos_path = os.path.join(dir_path, 'diff_map_intensity_thresh_pos.nii')
#                 difference_map_neg_path = os.path.join(dir_path, 'diff_map_intensity_thresh_neg.nii')

#                 if not os.path.exists(difference_map_pos_path):
#                     print(f'Applying positive intensity threshold to difference map for {sub_dir}')
#                     # Apply intensity threshold
#                     diff_map_intensity_thresholded_pos = apply_intensity_threshold(
#                         difference_map_path=difference_map_path, 
#                         intensity_threshold= 8.5,  # change this!
#                         cluster_type='pos')
#                     nib.save(diff_map_intensity_thresholded_pos, difference_map_pos_path)
#                 else:
#                     print(f'{difference_map_pos_path} already exists')
                    
#                 if not os.path.exists(difference_map_neg_path):
#                     print(f'Applying negative intensity threshold to difference map for {sub_dir}')
#                     # Apply intensity threshold
#                     diff_map_intensity_thresholded_neg = apply_intensity_threshold(
#                         difference_map_path=difference_map_path, 
#                         intensity_threshold= intensity_threshold, 
#                         cluster_type='neg')
#                     nib.save(diff_map_intensity_thresholded_neg, difference_map_neg_path)
#                 else:
#                     print(f'{difference_map_neg_path} already exists')
                    
###############################################################################################################################################
# Step 12: Apply volume threshold to difference maps
# Conda activate dess
###############################################################################################################################################
# print('Applying volume threshold to difference maps')

# intensity_threshold = 8.5 #in ms
# volume_threshold = 75 # number of voxels

# t2c_types = ['pos', 'neg']

# for t2c_type in t2c_types:
#     for sub_dir in os.listdir(data_path):
#         sub_dir_path = os.path.join(data_path, sub_dir)
        
#         # session_dirs= os.listdir(sub_dir_path)
        
#         idx_dirs= [dir for dir in os.listdir(sub_dir_path) if 'IDX' in dir]
#         cl_dirs= [dir for dir in os.listdir(sub_dir_path) if 'CL' in dir]
        
#         knee_dirs = [idx_dirs, cl_dirs]
        
#         for knee_dir in knee_dirs:
#             for session_dir in knee_dir:
#                 if '6mo' in session_dir:
#                     dir_path = os.path.join(sub_dir_path, session_dir)       
#                     difference_map_path = os.path.join(dir_path, f'T2C_int{intensity_threshold}_{t2c_type}.nii')
#                     difference_map_vol_path = os.path.join(dir_path, f'T2C_int{intensity_threshold}_vol{volume_threshold}_{t2c_type}.nii')
                    
#                     if not os.path.exists(difference_map_vol_path):
#                         print(f'Applying volume threshold to intensity thresholded {t2c_type} difference map for {sub_dir}')
#                         # Apply intensity threshold
#                         diff_map_volume_thresholded = apply_volume_threshold(
#                             difference_map_path=difference_map_path, 
#                             volume_threshold= volume_threshold
#                             )
#                         nib.save(diff_map_volume_thresholded, difference_map_vol_path)
#                     else:
#                         print(f'{difference_map_vol_path} already exists')
                    
                    
###############################################################################################################################################
# Step 13: Tabulate T2C metrics
# Conda activate dess
###############################################################################################################################################
# print('Tabulating T2C metrics')

# data_save_path = os.path.join(results_path, 'T2C_metrics.xlsx')

# data= pd.DataFrame()

# intensity_threshold = 8.5 #in ms
# volume_thresholds = [25, 75] # number of voxels
# t2c_types = ['pos', 'neg']

# for t2c_type in t2c_types:
    
#     for volume_threshold in volume_thresholds:

#         for sub_dir in os.listdir(data_path):
#             sub_dir_path = os.path.join(data_path, sub_dir)
            
#             # session_dirs= os.listdir(sub_dir_path)
            
#             idx_dirs= [dir for dir in os.listdir(sub_dir_path) if 'IDX' in dir]
#             cl_dirs= [dir for dir in os.listdir(sub_dir_path) if 'CL' in dir]
            
#             knee_dirs = [idx_dirs, cl_dirs]
            
#             for knee_dir in knee_dirs:
#                 for session_dir in knee_dir:
#                     if '6mo' in session_dir:
#                         print(session_dir)
#                         dir_path = os.path.join(sub_dir_path, session_dir)
#                         difference_map_path = os.path.join(dir_path, 'difference_map.nii')    
#                         t2c_path = os.path.join(dir_path, f'T2C_int{intensity_threshold}_vol{volume_threshold}_{t2c_type}.nii')
                        
#                         sub = session_dir.split('_')[0]
#                         knee= session_dir.split('_')[1]
                        
#                         if os.path.exists(difference_map_path) and os.path.exists(t2c_path):
#                             # Compute T2C metrics
#                             print(f'Computing {t2c_type} T2C metrics for {sub} {knee}')
#                             data_sub = compute_T2C_metrics(
#                                 cluster_map_path= t2c_path,
#                                 difference_map_path = difference_map_path
#                                 )
                            
#                             data_sub.insert(0, 'Intensity Threshold (ms)', intensity_threshold)
#                             data_sub.insert(1, 'Volume Threshold (voxels)', volume_threshold)
#                             data_sub.insert(2, 'Subject', sub)
#                             data_sub.insert(3, 'Knee', knee)
#                             data_sub.insert(4, 'Visit (months)', 6)

#                             data= pd.concat([data, data_sub], axis=0, ignore_index=True)
#                             append_df_to_excel(data= data_sub, data_file_path= data_save_path, sheet=f'{t2c_type}')
                            
#                         else:
#                             print(f'{difference_map_path} or {t2c_path} does not exist')       
                        
                        

###############################################################################################################################################                    
###############################################################################################################################################  

# Femoral Cartilage sub-regional analysis.

# Step 1: Subdivide the femoral cartilage into 5 subregions: Anterior, Medial and Lateral Weightbearing, and Medial and Lateral Posterior
# Step 2: Assign T2C to respective subregions based on minimum centroid-distance method

###############################################################################################################################################  
###############################################################################################################################################  

###############################################################################################################################################
# Step 1: Subdivide the femoral cartilage into 5 subregions: Anterior, Medial and Lateral Weightbearing, and Medial and Lateral Posterior
# Conda activate pymskt
###############################################################################################################################################
# print('Dividing femraol cartilage into 5 subregions')

# for sub_dir in os.listdir(data_path):
#     sub_dir_path = os.path.join(data_path, sub_dir)
    
#     # session_dirs= os.listdir(sub_dir_path)
    
#     idx_dirs= [dir for dir in os.listdir(sub_dir_path) if 'IDX' in dir]
#     cl_dirs= [dir for dir in os.listdir(sub_dir_path) if 'CL' in dir]
    
#     knee_dirs = [idx_dirs, cl_dirs]
    
#     for knee_dir in knee_dirs:
#         for session_dir in knee_dir:
#             if '6mo' in session_dir:
#                 print(session_dir)
#                 dir_path = os.path.join(sub_dir_path, session_dir)
#                 seg_path = os.path.join(dir_path, 'segmentation.nii')  
#                 fc_subregions_save_path = os.path.join(dir_path, 'segmentation_fc_subregions.nii')
#                 fc_subregions = generate_fc_subregions(
#                     input_seg_file_path= seg_path, 
#                     output_seg_file_path= fc_subregions_save_path
#                     )

###############################################################################################################################################
# Step 2: Assign T2C to respective subregions based on minimum centroid-distance method
# conda activate dess
###############################################################################################################################################

# print('Assign T2C to respective subregions based on minimum centroid-distance method')

# intensity_threshold = 8.5 #in ms
# volume_threshold = 75 # number of voxels

# data_save_path = os.path.join(results_path, f'T2C_metrics_subregions.xlsx')
                        
# t2c_types = ['pos', 'neg']

# for t2c_type in t2c_types:
    
#     data_save_path = os.path.join(results_path, f'T2C_metrics_subregions_{t2c_type}.xlsx')

#     for sub_dir in os.listdir(data_path):
#         sub_dir_path = os.path.join(data_path, sub_dir)
        
#         idx_dirs= [dir for dir in os.listdir(sub_dir_path) if 'IDX' in dir]
#         cl_dirs= [dir for dir in os.listdir(sub_dir_path) if 'CL' in dir]
        
#         knee_dirs = [idx_dirs, cl_dirs]
        
#         for knee_dir in knee_dirs:
#             for session_dir in knee_dir:
#                 if '6mo' in session_dir:
#                     print(session_dir)
#                     dir_path = os.path.join(sub_dir_path, session_dir)
#                     seg_path = os.path.join(dir_path, 'segmentation.nii')  
#                     fc_subregions_path = os.path.join(dir_path, 'segmentation_fc_subregions.nii')
#                     cluster_map_path = os.path.join(dir_path, f'T2C_int{intensity_threshold}_vol{volume_threshold}_{t2c_type}.nii')
#                     t2c_subregional_map_save_path = os.path.join(dir_path, f'T2C_int{intensity_threshold}_vol{volume_threshold}_subregions_{t2c_type}.nii')
                    
#                     if os.path.exists(cluster_map_path) == True and os.path.exists(fc_subregions_path) == True:
                        
#                         if os.path.exists(t2c_subregional_map_save_path) == False:
                            
#                             # get the cluster and subregions data
#                             cluster_com_data_sub, \
#                             subregions_com_data_sub, \
#                             t2c_subregion_distances_all_data_sub, \
#                             t2c_in_subregion_data_sub, \
#                             t2c_as_subregion_labels_img= assign_t2c_to_subregions(cluster_map_path, fc_subregions_path)
                            
#                             nib.save(t2c_as_subregion_labels_img, t2c_subregional_map_save_path)
                            
#                             columns_to_add = [
#                                 ('Intensity Threshold (ms)', intensity_threshold, 0),
#                                 ('Volume Threshold (voxels)', volume_threshold, 1),
#                                 ('Subject', session_dir.split('_')[0], 2),
#                                 ('Knee', session_dir.split('_')[1], 3),
#                                 ('Visit', 6, 4)
#                             ]
                            
#                             dataframes = [
#                             cluster_com_data_sub,
#                             subregions_com_data_sub,
#                             t2c_subregion_distances_all_data_sub,
#                             t2c_in_subregion_data_sub
#                             ]

#                             for df in dataframes:
#                                 for column_name, value, position in columns_to_add:
#                                     df.insert(position, column_name, value)
                            
#                             append_df_to_excel(data=cluster_com_data_sub, data_file_path= data_save_path, sheet= 'T2C COM')      
#                             append_df_to_excel(data=subregions_com_data_sub, data_file_path= data_save_path, sheet= 'Sub-regions COM')      
#                             append_df_to_excel(data=t2c_subregion_distances_all_data_sub, data_file_path= data_save_path, sheet= 'Distance bw T2C-Subregions COM')      
#                             append_df_to_excel(data=t2c_in_subregion_data_sub, data_file_path= data_save_path, sheet= 'T2C in sub-regions')      


            

            
