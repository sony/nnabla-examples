# JSIGAN: Joint Super-Resolution and Inverse Tone-Mapping
This repository contains the code (in NNabla) for "[JSI-GAN: GAN-Based Joint Super-Resolution and Inverse Tone-Mapping with Pixel-Wise Task-Specific Filters for UHD HDR Video](https://arxiv.org/abs/1909.04391)"
paper by [Soo Ye Kim et al](https://github.com/JihyongOh/JSI-GAN).

## Introduction
This paper focuses on divide-and-conquer approach in designing a novel GAN based joint SR-ITM network, called JSI-GAN, which is composed of three task-specific subnets:
* An image reconstruction (IR) subnet- reconstructs coarse HR HDR image.
* A detail restoration (DR) subnet - restores details to be added on the coarse image.
* And a local contrast enhancement (LCE) subnet - restores the local contrast.


## Prerequisites
* nnabla >= 1.10.0

## Inference
The pre-trained weights can be downloaded from the links provided in the below table:

### Pre-trained Weights
 | | JSINet | JSIGAN |
|---|---|---|
|Scaling Factor 1|[JSINet_X1](https://nnabla.org/pretrained-models/nnabla-examples/GANs/jsigan/jsinet_x2.h5)|[JSIGAN_x1](https://nnabla.org/pretrained-models/nnabla-examples/GANs/jsigan/jsigan_x1.h5)|
|Scaling Factor 2|[JSINet_X2](https://nnabla.org/pretrained-models/nnabla-examples/GANs/jsigan/jsinet_x2.h5)|[JSIGAN_x2](https://nnabla.org/pretrained-models/nnabla-examples/GANs/jsigan/jsigan_x2.h5)|
|Scaling Factor 4|[JSINet_X4](https://nnabla.org/pretrained-models/nnabla-examples/GANs/jsigan/jsinet_x4.h5)|[JSIGAN_x4](https://nnabla.org/pretrained-models/nnabla-examples/GANs/jsigan/jsinet_x4.h5)|

### Inference using the downloaded pre-trained weights.
Clone the nnabla-examples [repository](https://github.com/sony/nnabla-examples.git) and download the [test dataset](https://drive.google.com/file/d/1dZTwvRhf189L7NLkAcpij4980fyEXq3Q/view?usp=sharing)

Run the following command with scaling factor 2 or 4 to generate High resolution HDR images. Running the same command with scaling factor 1 will generate only HDR images without super-resolution.
```
python inference.py \
     --lr-sdr-test {path to lr sdr test dataset} \
     --hr-hdr-test {path to hr hdr test dataset} \
     --pre-trained-model {path to pre trained model} \
     --scaling-factor {1, 2 or 4} \
     --save-images {save predicted images if True. Set it to False to save time}
```
## Dataset preparation
The training dataset has been provided by the original authors and can be downloaded from [here](https://drive.google.com/file/d/19cp91wSRSrOoEdPeQkfMWisou3gJoh-7/view?usp=sharing). The dataset is in `.mat` format and it contains 39,840 frames in total taken from 4K-UHD HDR YouTube Videos. The SDR video pairs have been obtained from automatic conversion process of YouTube, and have been down scaled with bicubic filters.

### Inverse Tone-Mapping Dataset 
To train JSIGAN to only do inverse tone-mapping without super-resolution, we need a LR-HDR dataset. We can convert the existing HR-HDR dataset, HDR_youtube_80.mat(from train folder) to a LR-HDR dataset using the below script.

```
python generate_lr_hdr.py --hr-hdr-path {path to HR HDR images} \
                          --lr-hdr-path {path to save generate LR HDR images} \
	                  --reduction-factor {2} 
```

To generate a LR-HDR test dataset, convert HDR_youtube_80.mat from test folder and run the same script with create_test_dataset flag.
```
python generate_lr_hdr.py --hr-hdr-path {path to HR HDR images} \
                          --lr-hdr-path {path to save generate LR HDR images} \
	                  --reduction-factor {2} \
			  --create-test-dataset
```

## Training
JSIGAN training is divided in two steps:
1. A JSINet (Joint Inverse tone mapping and Super Reolution without GANs) model is trained with MSE loss on generator network.
2. Later the JSIGAN network having one generator and two discriminators, is finetuned using the pre-trained JSINet model. 
### Training a JSInet model 
Use the below code to start the training.
#### Single GPU training
```
python train.py \
     --lr-sdr-train {path to LR SDR images} \
     --hr-hdr-train {path to HR HDR images} \
     --output-dir {path to save trained model} \
     --pre-trained-model {path to pre trained model} \
     --scaling-factor {1, 2 or 4} 
```
#### Distributed Training
For distributed training [install NNabla package compatible with Multi-GPU execution](https://nnabla.readthedocs.io/en/latest/python/pip_installation_cuda.html#pip-installation-distributed). Use the below code to start the distributed training.
```
export CUDA_VISIBLE_DEVICES=0,1,2,3 {device ids that you want to use}
mpirun -n {no. of devices} python train.py \
     --lr-sdr-train {path to LR SDR images} \
     --hr-hdr-train {path to HR HDR images} \
     --output-dir {path to save trained model} \
     --pre-trained-model {path to pre trained model} \
     --scaling-factor {1, 2 or 4} 
```
### Training JSIGAN
JSIGAN, incorporates a novel detail loss so that generator mimics the realistic details in the ground truth and a feature-matching loss that helps in stabilizing the training process.The pre-trained JSINet model is used for finetuning the JSIGAN network.
To obtain this you can train a JSINet network or use our [pre-trained JSINet weights.](#pre-trained-weights)
Use the code below to train JSIGAN:
#### Single GPU training
```
python train.py \
     --lr-sdr-train {path to LR SDR images} \
     --hr-hdr-train {path to HR HDR images} \
     --output-dir {path to save trained model} \
     --pre-trained-model {path to pre trained model} \
     --scaling-factor {1, 2 or 4} \
     --jsigan 
```
#### Distributed training
```
export CUDA_VISIBLE_DEVICES=0,1,2,3 {device ids that you want to use}
mpirun -n {no. of devices} python train.py \
     --lr-sdr-train {path to LR SDR images} \
     --hr-hdr-train {path to HR HDR images} \
     --output-dir {path to save trained model} \
     --pre-trained-model {path to pre trained model} \
     --scaling-factor {1, 2 or 4} \
     --jsigan 
```
