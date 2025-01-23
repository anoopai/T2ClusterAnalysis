# T2-Cluster-Analysis

# Installation

###### Fork and clone the repository to your machine

```
https://github.com/anoopai/T2ClusterAnalysis.git
cd T2ClusterAnalysis
```

###### Install Dependencies

* DOSMA (for bone, cartilage and meniscus segmentation)
* PYMSKT (Dividing femoral cartilage into subregion, 3D visualisations from 3D meshes)

###### DOSMA

```
# create environment
conda create -n cluster_analysis python=3.8
conda activate cluster_analysis

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
```

Download DOSMA weights to perform automatic bone and tissue segmentation

```
# to download DOSMA model weights, install huggingface API
pip install huggingface-hub
```

pip install huggingface_hub

Login

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
# Download Dosma weights from Huggingface by running the following script
# make sure you are in the main repository (/T2ClusterAnalysis) before you run the script
python download_dosma_weights.py
```

#### PYMSKT

```
cd dependencies

# clone repository
git clone https://github.com/gattia/pymskt.git
  
# move into directory
cd pymskt
  
# INSTALLING DEPENDENCIES
# Recommend pip becuase cycpd and pyfocusr are available on pypi (but not conda)
pip install -r requirements.txt

# IF USING PIP
pip install .
```

###### T2 Cluster Analysis

```
# make sure you are in the main repository (Path/to/T2ClusterAnalysis/) before you install
pip install -r requirements.txt
```
