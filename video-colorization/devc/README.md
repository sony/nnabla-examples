# Deep Exemplar-based Video Colorization 

This repository contains the code (in NNabla) for "[Deep Exemplar-based Video Colorization ](https://arxiv.org/abs/1906.09909)" 
paper by [Bo Zhang et al.](https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization).

## Introduction
Deep Exemplar-based Video Colorization is the first end-to-end network for video colorization. It introduces a recurrent framework that unifies the semantic correspondence and color propogation steps to achive temporal consistancy while remaining failthful to the reference image.

__Result Examples__\

| Input frames               | reference                  | Output |
| :--------------------: | :---------------------: | :---------------------: |





## Inference

### Pre-trained Weights
pre-trained weights can be downloaded from the following link. 
[add link]
### Inference using pre-trained weights provided by original authors

Type the below commands in terminal to generate colorized video from a given input frames:
```
python inference.py --input_path {path to the input frames} --ref_path {reference image} 
```
