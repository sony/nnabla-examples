# CenterNet NNabla

NNabla Implementation of CenterNet.

> Xingyi Zhou, Dequan Wang, Philipp Kr¨ahenb¨uhl. Objects as Points. [arXiv technical report 2019](https://arxiv.org/abs/1904.07850).

It includes both training and inference code. Currently it supports only Object Detection.
For the developer Readme, please read this [Developer README](./src/lib/README.md).

## Requirements

- NNabla >= 1.6.0
- CUDA >= 9.0
- CuDNN >= 7.6
- Python >= 3.6
- graphviz: `sudo apt-get install -y graphviz`


## Installation

### Using pre-built Docker images

You can pull a pre-built Docker image from Docker Hub.

```bash
# Change the cuda version if needed.
docker pull nnabla/nnabla-ext-cuda-multi-gpu:py37-cuda90-multi-gpu-ubuntu16-v1.5.0
```

### Building the Docker image

The [NNabla-example Dockerfile](https://github.com/sony/nnabla-examples/blob/master/Dockerfile) is supplied to build images with cuda support and cudnn v7.

```bash
docker build -t you_image_name -f Dockerfile .
```

After building the Docker image, create Docker container. Execute the following commands inside Docker container.

```bash
# Install opencv, gcc
apt-get update ; apt-get install -y libopencv-dev gcc

# Install necessary Python packages
pip install -r requirements.txt
```

## Demo

You can perform object detection by CenterNet with a pre-trained model as following.

```bash
python src/demo.py ctdet --dataset <coco or pascal> --arch <resnet or dlav0> --num_layers <number of layers> --checkpoint <path to *.h5 file> --demo <test_image.jpg> --gpus <gpu to use> --debug 1 --save_dir <path to output directory>
```

The argument `--checkpoint` specifies the pre-trained weight file which can be obtained by either [donwloading it](#pretrained-weights-and-benchmarks) or [training it yourself](#training). Note that ```dataset```,  ```arch``` and  ```num_layers``` parameters must match with loaded weights.

Set the ```debug``` parameter controls the outputs from the detector:
 * 0 for no output
 * 1 for bounding boxes over original image
 * 2 for bounding boxes over original image, bounding boxes over rescaled input image and heatmaps per class

The output image is produced at the directory specified by `--save_dir`.


## Dataset Preparation

For COCO, download the dataset from http://cocodataset.org/#home .

After downloading zip files containing datasets from the above link, you can extract the dataset as following.

```bash
ARCHIVE_ROOT=<directory containing the downloaded zip files>
TARGET_DIR=<directory where the dataset will be located>
for zipfile in train2017.zip val2017.zip test2017.zip \
    annotations_trainval2017.zip image_info_test2017.zip
do
    unzip $ARCHIVE_ROOT/$zipfile -d $TARGET_DIR
done
```


### PASCAL VOC

For PASCAL VOC, see the example script located on ```src/lib/tools/get_pascal_voc.sh```. It downloads the dataset, the annotations already in COCO format and merges them.

## Training

### Availble pre-trained backbone networks.
To use the pre-trained backbone weights, download the weight file corresponding to the model configuration and locate under the directory you specify by `pretrained_model_dir` in the `--train-config` file.

For example, if you want to use a pretrained weight file for DLAv0 with 34 layers for mixed precision training (NHWC memory layout), you can download it as following. It will locate the file in `weights/backbone` (default location of pretrained weights specified in YAML config files.).

```bash
mkdir -p weights/backbone
cd weights/backbone && \
    curl -L -O https://nnabla.org/pretrained-models/nnabla-examples/object-detection/ceneternet/backbone/resnet18_nhwc_imagenet.h5
```

| Arch. | Num. of layers | Pretrained parameters |
|:---:|:---:|:---:|
| ResNet | 18 | [NCHW](https://nnabla.org/pretrained-models/nnabla-examples/object-detection/ceneternet/backbone/resnet18_nchw_imagenet.h5) / [NHWC](https://nnabla.org/pretrained-models/nnabla-examples/object-detection/ceneternet/backbone/resnet18_nhwc_imagenet.h5) |
| DLAv0 | 34 | [NCHW](https://nnabla.org/pretrained-models/nnabla-examples/object-detection/ceneternet/backbone/dla34_nchw_imagenet.h5) / [NHWC](https://nnabla.org/pretrained-models/nnabla-examples/object-detection/ceneternet/backbone/dla34_nhwc_imagenet.h5) |

### Training

The following example shows how to run DLAv0 training for Pascal VOC dataset with 4 GPUs.

```bash
mpirun -n 4 python src/main.py ctdet \
    --data_dir <Path to Pascal VOC dataset>
    --train-config cfg/dlav0_34_pascal_fp.yaml \
    -o <path to output training results & logs>
```

See config files in `cfg` for more details such as configurations for dataset, batch size, mixed precision training, and learninge rate scheduler. (**Note**: mixed precision training doesn't work at this moment for some reason. Any contribution to fix the issue is welcome!)

For a single GPU training, you can run the following.

```bash
python src/main.py ctdet \
    --data_dir <Path to Pascal VOC dataset>
    --train-config cfg/dlav0_34_pascal_fp.yaml \
    -o <path to output training results & logs>
```

To specify a GPU to use, set `CUDA_VISIBLE_DEVICES=<gpu id>` as an environment variable. For example;

```bash
CUDA_VISIBLE_DEVICES=1 python src/main.py ...(arguments continue)...
```

## Validation

You can use the ```test.py``` script for AP/mAP validation:

```bash
python src/test.py ctdet --dataset <coco or pascal> --data_dir <coco or pascal root folder> --arch <resnet or dlav0> --num_layers <number layers> --checkpoint <path to checkpoint file> --gpus <gpu to use>
```

You can also recalculate the AP .txt files for a series using ```test.py```:

```bash
python src/test.py ctdet --dataset <coco or pascal> --data_dir <coco or pascal root folder> --arch <resnet or dlav0> --num_layers <number layers> --checkpoint_dir <root folder of checkpoints> --gpus <gpu to use>
```

## Pretrained weights and benchmarks

The hyperparameters used for mixed precision and full precision training were the same as the ones used by the Pytorch repository.

Hyperparameter search was not made thoroughly so almost 1-2AP variance is expected. Doubling the number of epochs with large learning rate increase the AP by around 2-3 points.
 
The evaluation scripts from the official datasets is used to calculate AP/mAP. 

### PascalVOC

| Arch. | GPUs | MP*1 | LR | Epochs | LR steps | Batch size per GPU | mAP | Pretrained parameters (Click to download) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ResNet18 |  1 x V100 | No | 1.25e-4 | 70 | 45,60 | 32 | 68.40  | [Download](https://nnabla.org/pretrained-models/nnabla-examples/object-detection/ceneternet/ctdet/resnet_18_pascal_fp.h5)
| ResNet18 |  1 x V100 | Yes | 1.25e-4 | 70 | 45,60 | 32 | 69.24  | [Download](https://nnabla.org/pretrained-models/nnabla-examples/object-detection/ceneternet/ctdet/resnet_18_pascal_mp.h5)
| DLAv0 |  2 x V100 | No | 1.25e-4 | 70 | 45,60 | 16 | 74.72  | [Download](https://nnabla.org/pretrained-models/nnabla-examples/object-detection/ceneternet/ctdet/dlav0_34_pascal_fp.h5)
| DLAv0 |  1 x V100 | Yes | 1.25e-4 | 70 | 45,60 | 32 | 75.06 | [Download](https://nnabla.org/pretrained-models/nnabla-examples/object-detection/ceneternet/ctdet/dlav0_34_pascal_mp.h5)

### COCO


| Arch. | GPUs | MP*1 | LR | Epochs | LR steps | Batch size per GPU | AP (0.5-0.95 IoU)| Pretrained parameters (Click to download) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ResNet18 |  4 x V100 | No | 5.e-4 | 140 | 90,120 | 32 | 25.18 | [Download](https://nnabla.org/pretrained-models/nnabla-examples/object-detection/ceneternet/ctdet/resnet_18_coco_fp.h5)
| ResNet18 |  4 x V100 | Yes | 5.e-4 | 140 | 90,120 | 32 | 24.81 | [Download](https://nnabla.org/pretrained-models/nnabla-examples/object-detection/ceneternet/ctdet/resnet_18_coco_mp.h5)
| DLAv0 |  8 x V100 | No | 5e-4 | 140 | 90,120 | 16 | 31.77  | [Download](https://nnabla.org/pretrained-models/nnabla-examples/object-detection/ceneternet/ctdet/dlav0_34_coco_fp.h5)
| DLAv0 |  4 x V100 | Yes | 5e-4 | 140 | 90,120 | 32 | 31.85  | [Download](https://nnabla.org/pretrained-models/nnabla-examples/object-detection/ceneternet/ctdet/dlav0_34_coco_mp.h5)

## Export nnp file

`src/save_nnp.py` provides the way to export nnp file. Specify the `--dataset`, `--arch` and `--num_layers` options and the output will be saved in `exp/ctdet/your_exp_id/`.

```bash
# For example, to export DLAv034 nnp file, use the following command.
python src/save_nnp.py ctdet --dataset <coco or pascal> --arch dlav0 --num_layers 34
```

Please run `python src/save_nnp.py -h` to see which network is supported.

## Development/Extensions

### Adding new models

Implemented architectures are located on the ```src/lib/models/network```.
models are split between feature extractor and upsampling portion (ie:```dlav0_backbone.py``` and ```dlav0.py```).


### How to generate and update requirements.txt ?

requirements.txt is generated by [pip-compile command](https://github.com/jazzband/pip-tools). If you're going to use new Python package, first, add the package name to `install_requires` inside `setup.py`.

Update requirements.txt by running,

```bash
pip-compile
```

and you will get new requirements.txt.

## TODO List
* Add models with Deformable Convolution
* Add other backbones from paper (Hourglass101, etc)
* Add remaining tasks (keypoint extraction, 3D bounding box)
