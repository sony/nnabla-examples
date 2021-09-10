# Deep Exemplar-based Video Colorization 

This repository contains the inference code for "[Deep Exemplar-based Video Colorization ](https://arxiv.org/abs/1906.09909)" 
paper by [Bo Zhang et al.](https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization).

## Introduction
Deep Exemplar-based Video Colorization is claimed to be first end-to-end network for video colorization. It introduces a recurrent framework that unifies the semantic correspondence and color propagation steps to achieve temporal consistency while remaining faithful to reference image.

__Result Examples__

| Input               | Reference                  | Output |
| :--------------------: | :---------------------: | :---------------------: |
|![](images/examples/mov1.gif) | ![](images/examples/ref1.jpg) | ![](images/examples/out1.gif) | 
|![](images/examples/mov2.gif) | ![](images/examples/ref2.jpg) | ![](images/examples/out2.gif) |

## Quick Colorization Demo using DEVC

From the Colab link below, you can try using DEVC to generate colorized video from input frames. Please give it a try!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/deep-exemplar-based-video-colorization.ipynb)

## Getting started

### Prerequisites
* nnabla 
* cv2 >=3.3
* pillow
* numpy
* scikit-image

### Installation

For installation we recommend to use the [Anaconda](https://anaconda.org/) python distribution. To create a conda environment for _Deep Exemplar-based Video Colorization_, simply run:

`conda env create -f environment-X.yml` where `X` is either [`cpu`, `gpu`], depending on your system. 

## Inference

### Pre-trained Weights
Pre-trained weights can be downloaded from [here](https://nnabla.org/pretrained-models/nnabla-examples/deep-exemplar-based-video-colorization/devc_weights.zip).

Run the below command to unzip the downloaded file:  
`unzip devc_weights.zip`
### Inference using pre-trained weights provided by original authors
Run the below inference command to generate colorized video from given input frames in images/input/v32 directory:

```
python inference.py --input_path images/input/frames --ref_path images/ref/images --output_path images/output/ -c cudnn 
```
Colorized videos corresponding to the number of reference images will be generated in the output folder.    
#### Arguments:  
|Arguments  | Description | 
| --- | --- |  
| --input_path |  Path to input frames to be colorized |   
| --ref_path |  Path to reference image(s)  |
|--output_path |  Path to output folder (A folder will be created for every reference image in this location) | 
|--context or -c |  Context (Extension modules : `cpu` or `cudnn`)  |

