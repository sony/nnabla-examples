# Denoising Diffusion Probabilistic Models

This is a reproduction of [Improved Denoising Diffusion Probabilistic Models](http://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf) implemented by nnabla.
The code structure is inspired by the [original author's implementation](https://github.com/openai/improved-diffusion).

※ 2022/05/25
We also support ADM architecture proposed in [Diffusion Models Beat GANs on Image Synthesis](https://papers.nips.cc/paper/2021/file/49ad23d1ec9fa4bd8d77d02681df5cfa-Paper.pdf).

<p align="center">
<img src='imgs/cifar10.png', width="200">
<img src='imgs/imagenet64x64.png', width="200">
</p>
<p align="center">
<img src='imgs/celebahq.png', width="400">
</p>
Figure: Generated image samples by models trained on cifar10 (32x32), imagenet_64x64, and CelebA-HQ(256x256). Note that all samples are resized for visualization.
</p>

## Instalation
To install dependencies, run as below:
```
# install dependencies
pip install -r requirements.txt

# install NEU
cd /path/to/nnabla-examples/
pip install -e .

# install nnabla_diffusion package
cd /path/to/nnabla-examples/diffusion-models
pip install -e .
```

## Generating images by pre-trained models
To generate images by pre-trained weights, you should download a yaml file for model config and a h5 file.
Please munualy download them from the following list.

| Model | Dataset | Resolution | download link | FID | Iterations |
| :---: | :---: | :---: | :---: | :---: |  :---: |
| Improved DDPM | Cifar10 | 32 x 32 | [config](https://nnabla.org/pretrained-models/nnabla-examples/diffusion-models/hydra/config_cifar10_32x32.yaml) / [weight](https://nnabla.org/pretrained-models/nnabla-examples/diffusion-models/cifar10_32/params.h5)| 3.49 | 500K |
| Improved DDPM | Imagenet | 64 x 64 | [config](https://nnabla.org/pretrained-models/nnabla-examples/diffusion-models/hydra/config_imagenet_64x64.yaml) / [weight](https://nnabla.org/pretrained-models/nnabla-examples/diffusion-models/imagenet_64/params.h5) | 19.80 | 500K |
| Improved DDPM | CelebA-HQ | 256 x 256 | [config](https://nnabla.org/pretrained-models/nnabla-examples/diffusion-models/hydra/config_celebAHQ_256x256.yaml) / [weight](https://nnabla.org/pretrained-models/nnabla-examples/diffusion-models/celebAHQ_256/params.h5) | 30.73 | 300K |
| ADM | Imagenet | 256 x 256 | [config](https://nnabla.org/pretrained-models/nnabla-examples/diffusion-models/hydra/config_ADM_imagenet_256x256.yaml) / [weight](https://nnabla.org/pretrained-models/nnabla-examples/diffusion-models/ADM_imagenet_256/params.h5) | 43.80 | 300K |
| ADM | CelebA-HQ | 256 x 256 | [config](https://nnabla.org/pretrained-models/nnabla-examples/diffusion-models/hydra/config_ADM_celebAHQ_256x256.yaml) / [weight](https://nnabla.org/pretrained-models/nnabla-examples/diffusion-models/ADM_celebAHQ_256/params.h5) | 27.69 | 300K |
| ADM | FFHQ | 256 x 256 | [config](https://nnabla.org/pretrained-models/nnabla-examples/diffusion-models/ADM_FFHQ_256/config_ADM_FFHQ256.yaml) / [weight](https://nnabla.org/pretrained-models/nnabla-examples/diffusion-models/ADM_FFHQ_256/ADM_FFHQ256_ema_param.h5) | 8.608 | 200K |


Note that the FID scores shown on the list are computed by 10K generated images against training data for all datasets except 50K generated images for cifar10 and FFHQ.

After downloading them, you can generate images as follows:
```bash
python generate.py generate.config=<your config file path> generate.h5=<your h5 file path>
```
You can see generated results in `./outs` directory as default.
Running inference with above pretrained models on a single nvidia A100 GPU takes about 1 min for cifar10, 3 min for Imagenet 64x64, and 30 min for CelebA-HQ, respectively. 
You can also specify the sampling interval for generation by `generate.respacing_step={x}`. In this case, only T / x sampling steps are performed and inference time will be x times faster.

We also support DDIM sampler that enabls deterministic sampling. If you would like to use it, try `generate.ddim=True` option.
For more details about DDIM, please see the [original paper](http://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf).


## Generating image through browser user interface
We also provide browser UI to generate image by pretrained DPM.

First, you need to install additional packages: 
```
pip install -r server_requirements.txt
```

After that, please specify your model path (h5 and config) in `config/yaml/config_inference_server.yaml` and do below to launch your local inference server.
```
python launch_inference_server.py
```

After launching the server, you can access it through browser.
When you launch it on your local PC, just access `localhost:50000`.
If you launch it on other server (be sure to be able to access it through http connection), please specify its address rather than localhost.

## Download data for training

### cifar-10
The data iterator for Cifar-10 dataset will automatically download the dataset.
All you have to do is specifying `cifar10` as dataset name like `python train.py dataset=cifar10`.

### imagenet
To download the original ILSVRC2012 dataset, please follow [the instruction for the imagenet classification examples](https://github.com/sony/nnabla-examples/tree/master/image-classification/imagenet#preparing-imagenet-dataset).
After downloading it, you will have `<your data dir>/ilsvrc2012/train` and `<your data dir>/ilsvrc2012/val`.
You can train your model on imagenet with 256x256 resolution by specifying your dataset path as `python train.py dataset=imagenet dataset.data_dir=<your data dir>`.
(You should specify the parent directory having `train` as sub-directory.)

### CelebA-HQ
To download CelebA-HQ dataset, please follow [the official github](https://github.com/tkarras/progressive_growing_of_gans#preparing-datasets-for-training).

After creating dataset, you will have the directory named `<your data dir>/celeba-hq-{resolution}/images` which has all images as jpg format.
Note that you should use the same resolution as or larger resolution than the one you would like to train (e.g. if you train your model with 256x256, you should use 256x256 or larger).

Then, you can train your model on CelebA-HQ by `python train.py dataset=celebahq dataset.data_dir <your data dir>/celeba-hq-{resolution}`.
(You should specify the parent directory having `images` as sub-directory.)




### FFHQ-256
To download FFHQ-256 dataset, we use [the following link](https://www.kaggle.com/datasets/xhlulu/flickrfaceshq-dataset-nvidia-resized-256px).

After downloading dataset, you will have the directory named `<your data dir>/ffhq-256/images` which has all images as jpg format.

Then, you can train your model on FFHQ-256 by `python train.py dataset=ffhq256 dataset.data_dir <your data dir>/ffhq-256`.
(You should specify the parent directory having `images` as sub-directory.)



## Training
We prepare the training scripts for cifar-10, imagenet, CelebA-HQ and FFHQ-256 dataset, respectively.
See `scripts/` directory for more detail.

Note that all scripts assume multi-GPU training with 4 GPU.
If you would like to train your model with a single GPU, please remove `mpirun -N 4` from each script.

As for the dataset path, `./data` is used as `dataset.data_dir` as default.
In other words, we assume `./data` is a symbolic link for the path we describe in `Download data for training` section above.
If you get some errors related to dataset, please check your dataset path is correct.


# DDPM Segmentation
A reproduction of [Label-Efficient Semantic Segmentation with Diffusion Models](https://openreview.net/forum?id=SlxSY2UZQT) implemented by nnabla.
The code structure is inspired by the [original author's implementation](https://github.com/yandex-research/ddpm-segmentation).

## Download data for training
To download Dataset, please follow [the official github](https://github.com/yandex-research/ddpm-segmentation#datasets) and download file.  
Then, unzip through  `tar -xzf datasets.tar.gz -C <your data dir>/datasetddpm`.

## Training
For FFHQ pre-trained diffusion model (.h5 file), you can download from [this link](https://nnabla.org/pretrained-models/nnabla-examples/diffusion-models/ADM_FFHQ_256/ADM_FFHQ256_ema_param.h5).  
For FFHQ pre-trained dataset ddpm model, you can download from [this link](https://nnabla.org/pretrained-models/nnabla-examples/diffusion-models/ADM_FFHQ_256/datasetddpm/nnabla/ffhq_gmulti.zip).  
For another categories, you have to train ADM respectively.
```
## Example of FFHQ256
python train_pixel_classifier.py datasetddpm.h5=<your h5 file path> datasetddpm.config=yaml/config_train_pixelclassifier.yaml datasetddpm.steps=[50,150,250] datasetddpm.dim=[256,256,8448] datasetddpm.training_path=<your data dir>/datasetddpm/ffhq_34/real/train  datasetddpm.testing_path=<your data dir>/datasetddpm/ffhq_34/real/test
```

default category was set to "ffhq_34", if you use another category, you have to change `datasetddpm.category=<category name>`.


**Available dataset names**: bedroom_28, ffhq_34, cat_15, horse_21, celeba_19, ade_bedroom_30