# Neural Network Libraries - Examples

This repository contains working examples of [Neural Network Libraries](https://github.com/sony/nnabla/).
Before running any of the examples in this repository, you must install the Python package for Neural Network Libraries. The Python install guide can be found [here](https://nnabla.readthedocs.io/en/latest/python/installation.html).

Before running an example, also run the following command inside the example directory, to install additional dependencies:

```
cd example_directory
pip install -r requirements.txt
```


## Docker workflow

* Our Docker workflow offers an easy installation and setup of running environments of our examples.
* [See this page](doc/docker.md).

## nnabla-examples-utils (neu)

`neu` can now be installed as a python package. It provides a wide range of utility functions. For installation and usage, check [utils](utils/)

## Interactive Demos

We have prepared interactive demos, where you can play around without having to worry about the codes and the internal mechanism. You can run it directly on [Colab](https://colab.research.google.com/) from the links in the table below.

### Vision: Generation, Enhancement, Animation
|Name  | Notebook           | Task  | Example                       |
|:---------------------------------:|:-------------:|:-----:|:------------:|
| [SLE-GAN](https://arxiv.org/abs/2101.04775) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/slegan.ipynb) | Image Generation |<a href="url"><img src="https://github.com/sony/nnabla-examples/raw/master/GANs/slegan/example.png" align="center" height="90" ></a>|
| [First Order Motion Model](https://arxiv.org/abs/2003.00196) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/fomm.ipynb) | Facial Motion Transfer |<a href="url"><img src="https://github.com/sony/nnabla-examples/raw/master/GANs/first-order-model/imgs/sample_fake.gif" align="center" height="90" ></a>|
| [Zooming Slow-Mo](https://arxiv.org/abs/2002.11616) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/zooming_slowmo.ipynb) | Video Super-Resolution |<a href="url"><img src="https://github.com/sony/nnabla-examples/raw/master/frame-interpolation/zooming-slow-mo/demo/original.gif" align="center" height="90"></a> <a href="url"><img src="https://github.com/sony/nnabla-examples/raw/master/frame-interpolation/zooming-slow-mo/demo/zooming_slomo.gif" align="center" height="90" ></a>|
| [StyleGAN2](https://arxiv.org/abs/1912.04958) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/stylegan2.ipynb) | Image Generation |<a href="url"><img src="https://github.com/sony/nnabla-examples/raw/master/GANs/stylegan2/images/sample.png" align="center" height="90"  ></a>|
| [Deep-Exemplar-based-Video-Colorization](https://arxiv.org/abs/1906.09909) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/deep-exemplar-based-video-colorization.ipynb) | Video Colorization | <a href="url"><img src="https://github.com/sony/nnabla-examples/raw/master/video-colorization/deep-exemplar-based-video-colorization/images/examples/mov1.gif" align="center" height="90"></a> <a href="url"><img src="https://github.com/sony/nnabla-examples/raw/master/video-colorization/deep-exemplar-based-video-colorization/images/examples/out1.gif" align="center" height="90" ></a>|
| [TecoGAN](https://arxiv.org/abs/1811.09393) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/tecogan.ipynb) | Video Super-Resolution |<a href="url"><img src="https://github.com/sony/nnabla-examples/raw/master/GANs/tecogan/results/cropped_lr_city.gif" align="center" height="90"></a> <a href="url"><img src="https://github.com/sony/nnabla-examples/raw/master/GANs/tecogan/results/cropped_sr_city.gif" align="center" height="90" ></a>|
| [ESR-GAN](https://arxiv.org/abs/1809.00219)       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/esrgan.ipynb) | Super-Resolution|<a href="url"><img src="https://github.com/sony/nnabla-examples/raw/master/GANs/esrgan/results/comic.png" align="center" height="90"></a> <a href="url"><img src="https://github.com/sony/nnabla-examples/raw/master/GANs/esrgan/results/comic_SR.png" align="center" height="90" ></a>|
| [Self-Attention GAN](https://arxiv.org/abs/1805.08318)       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/sagan.ipynb) | Image Generation|<a href="url"><img src="https://camo.githubusercontent.com/afa86e7d0ccdb2ba75c273070f8dc8d50bda5d2821ebacbf1fc40a2dc0059b7c/68747470733a2f2f626c6f672e6e6e61626c612e6f72672f77702d636f6e74656e742f75706c6f6164732f323031382f31322f30343131303033342f3030303938352d333030783330302e706e67" align="center" height="90" ></a> <a href="url"><img src="https://camo.githubusercontent.com/9f829b60f2b53bfb77111f55997e6ed0f160f8b04437c046754a32503f90a1ba/68747470733a2f2f626c6f672e6e6e61626c612e6f72672f77702d636f6e74656e742f75706c6f6164732f323031382f31322f30343131303034322f3030303933332d333030783330302e706e67" align="center" height="90" ></a>|
| [StarGAN](https://arxiv.org/abs/1711.09020) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/stargan.ipynb) | Image Translation |<a href="url"><img src="https://github.com/sony/nnabla-examples/raw/master/GANs/stargan/imgs/sample_black_haired_female.png" align="center" height="90" ></a><a href="url"><img src="https://github.com/sony/nnabla-examples/raw/master/GANs/stargan/imgs/sample_blond_haired_female.png" align="center" height="90" ></a>|
| [DCGAN](https://arxiv.org/abs/1511.06434) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla/blob/master/tutorial/dcgan_image_generation.ipynb) | Image Generation ||

### Vision: Recognition
|Name  | Notebook           | Task  | Example                       |
|:---------------------------------:|:-------------:|:-----:|:------------:|
| [CenterNet](https://arxiv.org/abs/1904.07850) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/centernet.ipynb) | Object Detection |<a href="url"><img src="https://blog.nnabla.org/wp-content/uploads/sites/2/2021/02/02093255/centernet_result_resized.jpg" align="center" height="90" ></a>|
| [PSMNet](https://arxiv.org/abs/1803.08669) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/psmnet.ipynb) | Stereo Depth Estimation |<a href="url"><img src="https://github.com/sony/nnabla-examples/raw/master/stereo-depth/PSMnet/results/0006.png" align="center" height="90" ></a> <a href="url"><img src="https://github.com/sony/nnabla-examples/raw/master/stereo-depth/PSMnet/results/depth_sceneflow.png" align="center" height="90"></a>|
| [Face Alignment Network](https://arxiv.org/abs/1703.07332) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/fan.ipynb) | Facial Keypoint Detection |<a href="url"><img src="https://github.com/sony/nnabla-examples/raw/master/facial-keypoint-detection/face-alignment/results/example1.png" align="center" height="90" ></a><a href="url"><img src="https://github.com/sony/nnabla-examples/raw/master/facial-keypoint-detection/face-alignment/results/example2.png" align="center" height="90" ></a>|
| [YOLO v2](https://arxiv.org/abs/1612.08242) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/yolov2.ipynb) | Object Detection ||
| [ResNet](https://arxiv.org/abs/1512.03385)/[ResNeXt](https://arxiv.org/abs/1611.05431)/[SENet](https://arxiv.org/abs/1709.01507) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/imagenet_classification.ipynb) | Image Classification |<a href="url"><img src="https://github.com/sony/nnabla-examples/raw/master/imagenet-classification/results/rn50-mp-cos90-loss.png" align="center" height="90" ></a>|

### Audio
|Name | Notebook           | Task  | Example                       |
|:---------------------------------:|:-------------:|:-----:|:------------:|
| [D3Net](https://arxiv.org/abs/2011.11844) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/ai-research-code/blob/master/d3net/music-source-separation/D3Net-MSS.ipynb) | Music Source Separation ||
| [X-UMX](https://arxiv.org/abs/2010.04228) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/ai-research-code/blob/master/x-umx/X-UMX.ipynb) | Music Source Separation |<a href="url"><img src="https://github.com/sony/ai-research-code/raw/master/x-umx/imgs/umx-network-vs-x-umx-network.png" align="center" height="90" ></a>|

### Machine Learning
|Name| Notebook           | Task  | Example                       |
|:---------------------------------:|:-------------:|:-----:|:------------:|
 [Out-of-Core training](https://arxiv.org/abs/2010.14109) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/out_of_core_training.ipynb) | Out-of-Core training |<a href="url"><img src="https://github.com/sony/ai-research-code/blob/master/out-of-core-training/imgs/overview.png?raw=true" align="center" height="90" ></a>|
| [MixUp](https://openreview.net/pdf?id=r1Ddp1-Rb) / [CutMix](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.pdf) / [VH-Mixup](https://arxiv.org/pdf/1805.11272.pdf) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/dataaugmentation.ipynb) | Data Augmentation |<a href="url"><img src="https://blog.nnabla.org/wp-content/uploads/sites/2/2020/04/07131002/mixuped_img.png" align="center" height="90" ></a> <a href="url"><img src="https://blog.nnabla.org/wp-content/uploads/sites/2/2020/04/07131130/cutmixed_img.png" align="center" height="90" ></a>|
| [Virtual Adversarial Training](https://arxiv.org/abs/1704.03976) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla/blob/master/tutorial/vat_semi_supervised_learning.ipynb) | Semi-Supervised Learning ||
| [SiameseNet](https://arxiv.org/abs/1606.09549) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla/blob/master/tutorial/siamese_feature_embedding.ipynb) | Feature Embedding ||
| [Variational Auto-encoder](https://arxiv.org/abs/1312.6114) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla/blob/master/tutorial/vae_unsupervised_learning.ipynb) | Unsupervised Learning ||

### eXplainable AI
|Name| Notebook           | Task  | Example                       |
|:---------------------------------:|:-------------:|:-----:|:------------:|
 [Grad-CAM](https://arxiv.org/abs/1610.02391) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/gradcam.ipynb) | Visualization |<a href="url"><img src="https://github.com/sony/nnabla-examples/raw/master/responsible_ai/gradcam/images/sample.png" align="center" height="90" ></a>|
 [SHAP](https://arxiv.org/abs/1901.10436) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/shap.ipynb) | Visualization |<a href="url"><img src="https://github.com/sony/nnabla-examples/raw/master/responsible_ai/shap/images/sample.png" align="center" height="90" ></a>|
 [Skin color (Masked Images)](https://arxiv.org/abs/1901.10436) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/face_evaluation_tutorial_plugin.ipynb) | Facial evaluation for skin color |<a href="url"><img src="https://github.com/sony/nnabla-examples/raw/master/responsible_ai/face_evaluation/Masked_Images.png" align="center" height="90" ></a>|

### Fairness of Machine Learning
|Name| Notebook           | Task  | Example                       |
|:---------------------------------:|:-------------:|:-----:|:------------:|
Introduction of Fairness Workflow Tutorial | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/gender_bias_mitigation_german_cc.ipynb) | Dataset/Model Bias Check and Mitigation by Reweighing |<a href="url"><img src="https://github.com/sony/nnabla-examples/raw/master/responsible_ai/gender_bias_mitigation/images/fairness.png" align="center" height="90" ></a>|

### Model Quantization
|Name| Notebook           | Task  | Example                       |
|:---------------------------------:|:-------------:|:-----:|:------------:|
|Post-training quantization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/quantization_tutorial.ipynb) | Post-training quantization ||
