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
python src/demo.py ctdet --config_file <config file> --trained_model_path <path to params*.h5 file> --demo <test_image.jpg> --gpus <gpu to use> --debug 1 --save_dir <path to output directory>
```

The argument `--config_file` will load the network architecture and dataset settings from the YAML file. Check `cfg/*.yaml` files for more details.

The argument `--trained_model_path` specifies the pre-trained weight file which can be obtained by either [donwloading it](#pretrained-weights-and-benchmarks) or [training it yourself](#training). Note that the parameters in the configuration file set by `--config_file` must match with loaded weights.

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

For PASCAL VOC, see the example script located on ```src/tools/get_pascal_voc.sh```. It downloads the dataset, the annotations already in COCO format and merges them.

## Training

### Availble pre-trained backbone networks.
To use the pre-trained backbone weights, download the weight file corresponding to the model configuration and locate under the directory you specify by `pretrained_model_dir` in the `--config_file` file.

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
    --config_file cfg/dlav0_34_pascal_fp.yaml \
    --save_dir <path to output training results & logs>
```

See config files in `cfg` for more details such as configurations for dataset, batch size, mixed precision training, and learninge rate scheduler. (**Note**: mixed precision training doesn't work at this moment for some reason. Any contribution to fix the issue is welcome!)

For a single GPU training, you can run the following.

```bash
python src/main.py ctdet \
    --data_dir <Path to Pascal VOC dataset>
    --config_file cfg/dlav0_34_pascal_fp.yaml \
    --save_dir <path to output training results & logs>
```

To specify a GPU to use, set `CUDA_VISIBLE_DEVICES=<gpu id>` as an environment variable. For example;

```bash
CUDA_VISIBLE_DEVICES=1 python src/main.py ...(arguments continue)...
```

### Resume training from the checkpoint

You can resume training from a specific checkpoint. (Checkpoint includes learning information like the optimizer, trained weights and etc. The learning information is saved in `checkpoint_{epoch_number}.json`. The program will read this file and resume training.)

The following example shows how to resume DLAv0 training for object detection task with COCO dataset.

```bash
# Resume training. Assume that the checkpoint is saved in temp/checkpoints
# Using 4 GPUs in this case.
mpirun -n 4 python src/main.py ctdet \
    --data_dir <Path to COCO dataset>
    --config_file cfg/dlav0_34_coco_fp.yaml \
    --save_dir temp/ \
    --resume-from <epoch number>
```

## Validation

You can use the ```test.py``` script for AP/mAP validation:

```bash
python src/test.py ctdet --config_file <config file> --data_dir <coco or pascal root folder> --trained_model_path <path to params*.h5 file> --gpus <gpu to use>
```

You can also recalculate the AP .txt files for a series using ```test.py```:

```bash
python src/test.py ctdet --config_file <config file> --data_dir <coco or pascal root folder> --trained_model_dir <root folder of params*.h5 files> --gpus <gpu to use>
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

`src/save_nnp.py` provides the way to export nnp file. The *.nnp output will be saved in `--save_dir` or `exp/ctdet_{dataset}_{arch}_{num_layers}_{timestamp}` by default.

```bash
# For example, to export DLAv034 nnp file for COCO dataset, use the following command.
python src/save_nnp.py ctdet --config_file cfg/dlav0_34_coco_fp.yaml
```

Please run `python src/save_nnp.py -h` to see which network is supported.

## Memory layout conversion

You may want to change the memory layout of trained parameters from NHWC (trained with `channel_last=True`) to NCHW and vice versa, for fine-tuning on differnt tasks for example. You may also want to the remove the 4-th channel in the first convolution which was padded to RGB input during training for speed advantage.

The following command converts a parameter file to a desired configuration.

```bash
python src/convert_parameter_format.py {input h5 file} {output h5 file} -m {layout either nchw or nhwc} -3
```

See options with `python src/convert_parameter_format.py -h`.

## Development/Extensions

### Adding new models

Implemented architectures are located on the ```src/lib/models/network```.
models are split between feature extractor and upsampling portion (ie:```dlav0_backbone.py``` and ```dlav0.py```).

## TODO List
* Add models with Deformable Convolution
* Add other backbones from paper (Hourglass101, etc)
* Add remaining tasks (keypoint extraction, 3D bounding box)
