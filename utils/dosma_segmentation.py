def dosma_segmentation(qdess_file_path, output_file_path, weights_path):            
            
    import os
    import traceback
    import numpy as np

    import dosma as dm
    from dosma.models import StanfordQDessUNet2D
    from dosma.scan_sequences import QDess
    
    if os.path.exists(qdess_file_path):
    
        qdess= QDess.load(qdess_file_path)
        echo1= qdess.volumes[0]
        echo2= qdess.volumes[1]
        echo1, echo2 = echo1.astype(float), echo2.astype(float)
        rss = np.sqrt(echo1 ** 2 + echo2 **2)

        # get input shape for the model
        input_shape = qdess.volumes[0].shape[:2] + (1,)

        # This model is 2D and currently requires the shape of the volume as an input for initialization
        model = StanfordQDessUNet2D(input_shape, weights_path)

        # The output is a dictionary of strings -> MedicalVolume
        # The keys are string ids for different tissues (fc, tc, pc, men)
        outputs = model.generate_mask(rss)

        # Save segmentations 
        tissues = ["all"]

        for tissue in tissues:
            segmentation = outputs[tissue].reformat_as(rss)
            segmentation.save_volume(output_file_path)
    
    else:
        print('Qdess file does not exist. Please provide valid file path')