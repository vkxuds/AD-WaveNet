# AD-WaveNet #
A network that integrates the Discrete 2D Wavelet Transform (DWT) with adaptive attention mechanisms to enhance sparse image reconstruction.

Abstract:

- - - -
## Network Architecture ##

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

- - - -
## Train ##

### Modify Dataset Path 
In `utils/dataset.py`, locate the line where the dataset path is specified (around line 130 and 131) and update it to the path of your local train dataset. 

In `train.py`, locate the line where the dataset path is specified (around line 145 and 146) and update it to the path of your local val dataset. 

### Training AD-WaveNet 
```
python train.py --lr 0.001 --outf logs/adwave_datasetname_lvl_2_date --lvl 2 --batchSize 128
```
- - - -
## Test ##
```
python test.py  --test_data test_dataset
```
coming for soon
