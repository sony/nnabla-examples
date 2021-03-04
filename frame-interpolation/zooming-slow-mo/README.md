# Zooming Slow-Mo: Fast and Accurate One-Stage Space-Time Video Super-Resolution.
This repository contains the code (in NNabla) for "[ Zooming Slow-Mo: Fast and Accurate One-Stage Space-Time Video Super-Resolution](https://arxiv.org/abs/2002.11616.pdf)"
paper by [Xiaoyu Xiang et al.](https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020)

## Introduction
This paper focuses on automatically generating a photo-realistic video sequence with a high space-time resolution from a low-resolution and low frame rate input video. 
Major contributions made in this paper are:
* Propose a single-stage space-time super-resolution network that can address temporal interpolation and spatial super-resolution simultaneously in a unified framework. This method is more effective than two-stage methods taking advantage of the intra-relatedness between the two sub-problems and also computationally more efficient.
* Propose a frame feature temporal interpolation network leveraging local temporal contexts based on deformable sampling for intermediate LR frames.
* Single-stage method achieves state-of-the-art STVSR performance on both Vid4 and Vimeo test data.

#### Result Examples
Original                   |   Zooming Slow-Mo         |   Only Slow-Mo
:-------------------------:|:-------------------------:|:-------------------------:
![orginal gif](demo/original.gif) |  ![Zooming Slo-Mo gif](demo/zooming_slomo.gif) | ![Only Slo-Mo gif](demo/onlyslomo.gif)

#### Evaluation Results
Evaluation has been done on *VID4 dataset*, which contains HR frames for 4 different categories.
Various evaluation metrics used in this paper are:<br>
PSNR: Pixel-wise accuracy<br>
SSIM: Structural similarity<br>
&#8593; : Stands for, the bigger, the better

| | PSNR &#8593;| SSIM &#8593;|
|---|---|---|
| Reported in the ZoomingSloMo paper | 26.31 | 0.7976 |
| ZoomingSloMo author's pretrained weights | 26.38 | 0.7977 |
| NNabla(This repo)* | 26.35 |0.7969 |

*The ZoomingSloMo model takes around 8 days for training on ` Nvidia GeForce RTX 2080 Ti` GPU.

## Prerequisites
* nnabla >= v1.17.0
* matplotlib
* seaborn

## Inference
The pre-trained ZoomingSloMo weights can be used to generate a high-resolution (HR) slow-motion video frames from a low frame rate (LFR), low-resolution (LR) video frames. The pre-trained weights can be downloaded from the links provided below:

### Pre-trained Weights
| Zooming-Slow-Mo weights | Slo-Mo weights |
|---|---|
|[Zooming-Slow-Mo pre-trained weights](https://nnabla.org/pretrained-models/nnabla-examples/frame-interpolation/zooming-slo-mo/zooming_slo_mo.h5)|[Slo-Mo pre-trained weights](https://nnabla.org/pretrained-models/nnabla-examples/frame-interpolation/zooming-slo-mo/slo_mo.h5)|
### Inference using the downloaded pre-trained weights.
Clone the nnabla-examples [repository](https://github.com/sony/nnabla-examples.git) and run the following commands
```
cd nnabla-examples/frame-interpolation/zooming-slow-mo
```
Run the following command to generate HR slow-motion video frames from a given low frame rate (LFR), low-resolution (LR) video frames
```
python inference.py --model {path to downloaded Zooming-Slo-Mo NNabla weight file} --input-dir {input directory}
```
Run the following command to generate slow-motion video frames from a given low frame rate (LFR) video frames
```
python inference.py --model {path to downloaded Slo-Mo NNabla weight file} --input-dir {input directory} --only-slomo
```

## Evaluation
Download Vid4 from [author's repo] (https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020/tree/master/datasets#vid4). 
And run the following command to calculate PSNR and SSIM
```
python inference.py --model {path to downloaded Zooming-Slo-Mo NNabla weight file} --input-dir {path to downloaded Vid4 (Vid4/LR/)} --metrics
```

## Dataset preparation
1. Download the original training + test set of `Vimeo-septuplet` (82 GB).
```Shell
wget http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip    
apt-get install unzip
unzip vimeo_septuplet.zip
```
**vimeo_septuplet/sequences**: The folder structure is as follows:
```
vimeo_septuplet
├── sequences
    ├── 00001
        ├── 0266
            ├── im1.png
            ├── ...
            ├── im7.png
        ├── 0268...
    ├── 00002...
├── readme.txt
├── sep_trainlist.txt
├── sep_testlist.txt
```
2. Generate low resolution(LR) images
```Shell
cd nnabla-examples/frame-interpolation/zooming-slow-mo
python authors_scripts/generate_lr.py \
       --dataset-path {path to vimeo dataset i.e `vimeo_septuplet/sequences`} \
       --scale 4
```
3. Create the LMDB files for faster I/O speed
```python
# Create LMDB dataset for GT frames
python authors_scripts/create_lmdb.py \
      --dataset-path {path to GT frames i.e `vimeo_septuplet/sequences`} \
      --save-lmdb-path {path to save lmdb files i.e `lmdb_data`} \
      --train-list-file {file containing list of training images i.e `vimeo_septuplet/sep_trainlist.txt`} \
      --mode GT

# Create LMDB dataset for LR frames
python authors_scripts/create_lmdb.py \
       --dataset-path {path to LR frames i.e `vimeo_septuplet/sequences_LR`} \
       --save-lmdb-path {path to save lmdb files i.e `lmdb_data`} \
       --train-list-file {file containing list of training images i.e `vimeo_septuplet/sep_trainlist.txt`} \
       --mode LR
```
The structure of generated lmdb folder is as follows:
```
lmdb_data
├── Vimeo7_train_GT.lmdb
    ├── data.mdb
    ├── lock.mdb
├── Vimeo7_train_LR.lmdb
    ├── data.mdb
    ├── lock.mdb
```
## Training ZoomingSloMo
Use the code below to train ZoomingSloMo network:
#### Single GPU training
```
python train.py \
     --batch-size 12 \
     --lmdb-data-gt {path to LMDB folder of GT images i.e `lmdb_data/Vimeo7_train_GT.lmdb`} \
     --lmdb-data-lq {path to LMDB folder of LR images i.e `lmdb_data/Vimeo7_train_LR.lmdb`} \
     --output-dir {path to save trained model} 
```
#### Distributed training
```
export CUDA_VISIBLE_DEVICES=0,1,2,3 {device ids that you want to use}
mpirun -n {no. of devices} python train.py \
     --batch-size 12 \
     --lmdb-data-gt {path to LMDB folder of GT images i.e `lmdb_data/Vimeo7_train_GT.lmdb`} \
     --lmdb-data-lq {path to LMDB folder of LR images i.e `lmdb_data/Vimeo7_train_LR.lmdb`} \
     --output-dir {path to save trained model} 
```

## Training SloMo
Use the code below to train SloMo network without super-resolution:
#### Single GPU training
```
python train.py \
     --batch-size 12 \
     --lmdb-data-gt {path to LMDB folder of GT images i.e `lmdb_data/Vimeo7_train_GT.lmdb`} \
     --lmdb-data-lq {path to LMDB folder of LR images i.e `lmdb_data/Vimeo7_train_LR.lmdb`} \
     --output-dir {path to save trained model} \
     --only-slomo 
```
