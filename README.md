# T2-Cluster-Analysis

# Installation

1. Fork and clone the repository to your machine
2. Install Dependencies

* DOSMA (for bone, cartilage and meniscus segmentation)
* pymskt (get 3D surfaces from 3D segmentations)

```
# create environment
conda create -n t2_cluster_analysis python=3.8
conda activate t2_cluster_analysis

# install non-python dependcies
conda install anaconda::cmake
conda install conda-forge::lit

# make directory to download and install dependencies
mkdir dependencies
cd dependencies

# Install DOSMA
git clone https://github.com/gattia/DOSMA
cd DOSMA
git checkout bone_seg
pip install '.[ai]'

# Install pymskt 
pip install mskt

# Install other modules/libraries/dependencies
pip install -r requirements.txt

# NOTE: IF HAVING ISSUES WITH TF, INSTALL VERSION 2.11 (pip install tensorflow==2.11)
```

Download DOSMA weights to perform automatic bone and tissue segmentation

```
huggingface-cli login
```

Once you run the above, you will then input your access token from huggingface. You will need to have/create a hugginface account, and then can get the access token by:

* login to huggingface
* click on user icon in top right
* click settings
* click "access tokens" on the left
* click "Create new token" on left (topish) - Modify Repositories Permission to allow Read/Interact/Write access.
* copy the token and input into the commandline prompt

```
# Download Dosma weights from Huggingface 
python download_dosma_weights.py
```
