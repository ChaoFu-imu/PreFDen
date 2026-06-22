
# Pre-trained Foundation Model for Uncorrelated Noise Attenuation of Seismic Data

[![](https://img.shields.io/badge/Dataset-v1.0-blue.svg)](https://champyin.com)![](https://img.shields.io/badge/python-3.11-green.svg?style=flat)

**Pre-trained Foundation Model** is a seismic denoising method built on foundation models with sparse attention networks. Pre-trained on abundant open seismic datasets, the model has strong generalization. In practical use, it only needs a small amount of data for quick fine-tuning. It cuts model parameters and improves engineering efficiency.

# :star:  Overview

## Current unsupervised denoising methods overlook signal sparsity, suffering from redundant parameters. This model adopts sparse attention and pre-training strategies and mainly includes four modules below.
 - Step 1: Data Preparation: Both synthetic seismic data and field seismic data are processed into patches, with zero-padding applied to boundary regions. In the pre-trained stage the dataset is split into training and validation sets at a ratio of 8:2. All data are converted to float32 format, and a TensorDataset is constructed for model training.
 - Step 2: Pre-training: We use a two-stage pre-training strategy. First, we train the model for several epochs on massive open datasets. The model roughly learns the features of seismic waves. We save the best training weights to provide better initial parameters for the next training step. Then, we train the model for another several epochs based on the saved weights. This step enables the model to learn detailed seismic wave features and output the final pre-trained weights.
 - Step 3: Fine-tuning:On the basis of the final pre-trained weights, we train the model on target datasets with only a small number of epochs.
 - Step 4:Inference and Validation: The optimal model weights obtained during training are loaded to perform patch-wise inference on noisy seismic data, and then the patched results are merged into complete denoised seismic data through inverse patch transformation. Quantitative indicators such as Signal-to-Noise Ratio (SNR) are calculated to conduct quantitative evaluation on the denoising effect of the model.` Median Filter`, `Pre-trained Foundation Model` and `PatchUnet` are adopted to test and compare the denoising performance.
--The trained network parameters are saved as `xx.pth` file
---
- `data`：All **data** is stored in this folder, including: `Amirsyn.mat`,`Amirsyn2d.mat`,`seis2dsyn.mat`, `seis3dsyn.mat`, `real2d.mat`, `real3d.mat`

|name     |data size|Patch size|Slid size|Patches|
| :--------|:--- | :----- | :-------- |:-------- |
|Amirsyn2d + Coh2d| 500 $\times$ 700   |   30 $\times$ 30   | 4 $\times$ 4   |23283 |
|Amirsyn| 500 $\times$ 35 $\times$ 40 |  12 $\times$ 12 $\times$ 12 | 4 $\times$ 2 $\times$ 2 |23985 |
|seis2dsyn| 512 $\times$ 64   |   30 $\times$ 30   | 1 $\times$ 1   |16905 |
|seis3dsyn| 401 $\times$ 64 $\times$ 64 |  12 $\times$ 12 $\times$ 12 | 3 $\times$ 1 $\times$ 1 |367979 |
|real2d   |512 $\times$ 128   |   30 $\times$ 30   | 1 $\times$ 1   |47817 |
|real3d   |256 $\times$ 64 $\times$ 48 |  12 $\times$ 12 $\times$ 12 | 3 $\times$ 1 $\times$ 1 |162763 |
#  :rocket:  File Description
- `Fine-tuning`：Both the denoising networks of Pre-trained Foundation Model during the fine-tuning stage for synthetic data and field data are stored here, including both 2D and 3D versions.
- `PatchUnet`：Both the denoising networks of PatchUnet for synthetic data and field data are stored here, including both 2D and 3D versions.
- `medianFilter`：Both the denoising networks of Median filter for synthetic data and field data are stored here, including both 2D and 3D versions.
 :boom: **Note Only one type of seismic data is presented here. If you are interested in other datasets, you can replace the corresponding input `.pth` model parameters and patch data  `D_patch` .** 
 
# The network structure of the pre-trained stage is shown below:
![Pre-trained](Fig/Pre-trained.png)
# The network structure of the fine-tuning stage is shown below:
![Fine-tuning](https://github.com/furser1/ISAnet/blob/main/Fig/1.png)
# :hotsprings: Example
## Loading the .mat Data 
- All data is stored in .mat format. You can run this script to convert the format from` MATLAB `->` Python`.
```makedown
import scipy.io as sio
data = sio.loadmat('syn2dsyn.mat')
Dc = data['DataClean']
Dn = data['DataNoisy']
```
##  Taking synthetic 2D seismic data as an example
- The figure below shows the clean data and the data contaminated by irrelevant noise. It contains one set of horizontal events and two sets of intersecting dipping events.
![syn2d](https://github.com/furser1/ISAnet/blob/main/Fig/3.png)

# :sunrise_over_mountains: Maintainer
Chao Fu

For any questions regarding the dataset, `pth` or scripts, please open an issue in this repository.
