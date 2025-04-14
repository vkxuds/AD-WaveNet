# AD-WaveNet #
A network that integrates the Discrete 2D Wavelet Transform (DWT) with adaptive attention mechanisms to enhance sparse image reconstruction.

Abstract: Photoacoustic tomography (PAT) provides high-contrast, high-resolution biomedical images at rapid speeds. However, the quality of these images is highly sensitive to sampling density. Sparse sampling can significantly reduce equipment costs but often leads to image artifacts and degraded quality. While deep learning models have greatly enhanced sparse PAT imaging, their high computational requirements limit their use in resource-constrained environments. To overcome this challenge, we propose AD-WaveNet, a lightweight network that integrates the Discrete 2D Wavelet Transform (DWT) with adaptive attention mechanisms. This approach enhances sparse image reconstruction while reducing computational complexity. The attention mechanisms are specifically designed to exploit the multi-scale decomposition properties of DWT, allowing the model to emphasize key features across various scales. Compared to the latest models, AD-WaveNet reduces computational complexity and parameter count by nearly two orders of magnitude, while maintaining optimal reconstruction quality. This demonstrates AD-WaveNet’s significant potential for practical applications in PAT imaging.

- - - -
## Network Architecture ##
![Image Description](img/adwave.png)

- - - -
## Installation ##
1.Make conda environment
```
conda create -n adwave python=3.7
```
2.Install dependencies
```
pip install -r requirements.txt
```
- - - -
## Dataset ##

The dataset can be accessed at https://drive.google.com/drive/folders/1mgtLv9YjlmMkseq3aK9NBoAClI1wHoMe?usp=drive_link

### Dataset Format

The publicly available dataset is provided in VTK format. Before use, it needs to be converted to TIFF format images. Please use the `vtk2tif.py` script to perform the conversion.

- - - -
## Train ##

### Modify Dataset Path 
In `utils/dataset.py`, locate the line where the dataset path is specified (around line 130 and 131) and update it to the path of your local train dataset. 

In `train.py`, locate the line where the dataset path is specified (around line 145 and 146) and update it to the path of your local val dataset. 

### Training AD-WaveNet 

```
python train.py --lr 0.001 --outf logs/adwave_datasetname_lvl_2_date --lvl 2 --batchSize 128
```
The appropriate patch size and number of levels(lvl) may vary depending on the data. Please try adjusting them to achieve the optimal performance.
- - - -
## Test ##
```
python test.py  --test_data test_input --test_true_data test_true --load_pth pretrain.pth
```

## 📄 Paper

This repository contains the official implementation of the paper:

**"Lightweight sparse optoacoustic image reconstruction via an attention-driven multi-scale wavelet network"**  
*Photoacoustics, 2025: 100695*  
[Link to the paper](https://doi.org/10.1016/j.pacs.2025.100695)

If you use this code or find it helpful, please consider citing our paper:

```bibtex
@article{zhao2025lightweight,
  title={Lightweight sparse optoacoustic image reconstruction via an attention-driven multi-scale wavelet network},
  author={Zhao, Xudong and Hu, Shiqi and Yang, Qianqian and others},
  journal={Photoacoustics},
  year={2025},
  pages={100695}
}
```
## Acknowledgements 
﻿
This project is based on the [WINNet](https://github.com/jjhuangcs/WINNet) created by [jjhuangcs]. We have utilized its code framework and made significant modifications to adapt it to our use case.
