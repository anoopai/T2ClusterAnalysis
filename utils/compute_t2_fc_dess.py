def compute_t2_fc(qdess_path, mask_path, t2_save_path):

    import dosma as dm
    import os
    import SimpleITK as sitk

    import numpy as np
    import matplotlib.pyplot as plt

    from dosma import preferences
    from dosma.scan_sequences import QDess
    from dosma.tissues import FemoralCartilage
    from dosma.core.quant_vals import T2


    # Load qdess scan for baseline
    qdess= QDess.load(qdess_path)

    # load the segmentation for femoral cartilage
    reader = dm.NiftiReader()
    mask = reader.load(mask_path)

    # Define a femoral cartilage object.
    fc = FemoralCartilage()
    fc.set_mask(mask) 
        
    print('Computing T2 map...')

    # Generate t2 map, alpha (flip anlge)=25 from Dr.Balck's thesis
    t2= qdess.generate_t2_map(fc, tr= 0.01766e3, te= 0.005924e3, tg= 0.001904e6, alpha= 20, gl_area=3132)
        
    # Clip the estimated T2 values between [0, 80]
    t2.volumetric_map = np.clip(t2.volumetric_map, 0, 80)

    # Convert the np.array into float64 from int16
    t2.volumetric_map = t2.volumetric_map.astype(np.float64)
    
    # Convert the t2 map to sitk image
    # sitk_t2map = t2.volumetric_map.to_sitk(image_orientation='sagittal')
    # sitk.WriteImage(sitk_t2map, t2_save_path)
    
    # save the t2 map
    t2.volumetric_map.save_volume(t2_save_path)
    