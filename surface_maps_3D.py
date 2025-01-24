from utils.create_3D_surface import create_3D_surace

# Enter the path to the T2C Map, T2 difference map, etc (nii format)
map_path= '/dataNAS/people/anoopai/T2ClusterAnalysis/data/10-P/VISIT-5/aclr/results/t2c_subregions.nii.gz'

# Enter the path to the corresponding tissue femoral cartilage segmentation file (nii format)
seg_path = '/dataNAS/people/anoopai/T2ClusterAnalysis/data/10-P/VISIT-5/aclr/results/seg2_fc_reg.nii.gz'

# Create the 3D surface and save it to a .vtk file
print('Creating 3D surface from segmentation and map')
create_3D_surace(map_path, seg_path)


