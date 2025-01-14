def dosma_segmentation_bone_cartilage(qdess_file_path, output_file_path, weights_path):            
            
    import os
    import numpy as np

    import dosma as dm
    from dosma import file_constants
    from dosma import ImageDataFormat
    from dosma.models import StanfordQDessUNet2D
    from dosma.scan_sequences import QDess, CubeQuant, Cones
    from dosma.tissues import FemoralCartilage
    from dosma.models import StanfordQDessBoneUNet2D, StanfordCubeBoneUNet2D 
        
        
    if os.path.exists(qdess_file_path):
    
        qdess= QDess.load(qdess_file_path)
        qdess_rss= qdess.calc_rss()

        model_class = StanfordQDessBoneUNet2D
        model = model_class(weights_path, orig_model_image_size=(512,512))

        output = model.generate_mask(qdess_rss)
        output['all']= output['all'].astype(float)
        
        segmentation = output['all'].reformat_as(qdess_rss)
        segmentation.save_volume(output_file_path)
        
    else:
        print(f'{qdess_file_path} does not exist')
