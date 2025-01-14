# # in python script: 
# from huggingface_hub import snapshot_download
# snapshot_download(repo_id="aagatti/ShapeMedKnee", local_dir='./NSM_MODELS')

# in python script: 
from huggingface_hub import snapshot_download
import os

os.chdir('./files')
snapshot_download(repo_id="aagatti/dosma_bones", local_dir='dosma_weights')