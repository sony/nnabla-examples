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


## Interactive Demos

We have prepared interactive demos, where you can play around without having to worry about the codes and the internal mechanism. You can run it directly on [Colab](https://colab.research.google.com/) from the links in the table below.


| Name        | Notebook           | Task  |
|:------------------------------------------------:|:-------------:|:-----:|
| [ESR-GAN](https://arxiv.org/abs/1809.00219)       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/esrgan.ipynb) | Super-Resolution|
| [Self-Attention GAN](https://arxiv.org/abs/1805.08318)       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/sagan.ipynb) | Image Generation|
| [Face Alignment Network](https://arxiv.org/abs/1703.07332) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/fan.ipynb) | Facial Keypoint Detection |
| [PSMNet](https://arxiv.org/abs/1803.08669) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/psmnet.ipynb) | Depth Estimation |
| [ResNet](https://arxiv.org/abs/1512.03385)/[ResNeXt](https://arxiv.org/abs/1611.05431)/[SENet](https://arxiv.org/abs/1709.01507) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/imagenet_classification.ipynb) | Image Classification |
| [YOLO v2](https://arxiv.org/abs/1612.08242) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/yolov2.ipynb) | Object Detection |
| [StarGAN](https://arxiv.org/abs/1711.09020) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/stargan.ipynb) | Image Translation |
| [MixUp](https://openreview.net/pdf?id=r1Ddp1-Rb) / [CutMix](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.pdf) / [VH-Mixup](https://arxiv.org/pdf/1805.11272.pdf) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/dataaugmentation.ipynb) | Data Augmentation |
| [StyleGAN2](https://arxiv.org/abs/1912.04958) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/stylegan2.ipynb) | Image Generation |
| [X-UMX](https://arxiv.org/abs/2010.04228) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/ai-research-code/blob/master/x-umx/X-UMX.ipynb) | Music Source Separation |
| [DCGAN](https://arxiv.org/abs/1511.06434) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla/blob/master/tutorial/dcgan_image_generation.ipynb) | Image Generation |
| [Virtual Adversarial Training](https://arxiv.org/abs/1704.03976) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla/blob/master/tutorial/vat_semi_supervised_learning.ipynb) | Semi-Supervised Learning |
| [Variational Auto-encoder](https://arxiv.org/abs/1312.6114) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla/blob/master/tutorial/vae_unsupervised_learning.ipynb) | Unsupervised Learning |
| [SiameseNet](https://arxiv.org/abs/1606.09549) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla/blob/master/tutorial/siamese_feature_embedding.ipynb) | Feature Embedding |
| [Out-of-Core training](https://arxiv.org/abs/2010.14109) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/out_of_core_training.ipynb) | Out-of-Core training |
