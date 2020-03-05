# CenterNet NNabla

NNabla Implementation of CenterNet. Includes train codes. Currently supports only Object Detection.

For the developer Readme, please read this [Developer README](https://github.com/nnabla/nnabla-examples/blob/feature/20200305-centernet-mixed-precision/object-detection/centernet/src/lib/README.md).

## Supported backends

 - ResNet
 
 TODO add weights URL 

 - DLA34
 
TODO add weights URL

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

## Dataset Preparation

For COCO, download the dataset from http://cocodataset.org/#home .

### PASCAL VOC

For PASCAL VOC, use the script located on ```src/tools/get_pascal_voc.sh```. It downloads the dataset, the annotations already in COCO format and merges them.

```bash
cd src/tools
./get_pascal_voc.sh
```

The dataset will be generated to `src/tools/voc/` folder.

## Training

For training, run the following:

```bash
python main.py ctdet --dataset <coco or pascal> --arch <res or dlav0> --num_layers <18 or 34> --batch_size <batch size> --gpus <gpu to use>
```

For resuming, add the argument ```--checkpoint <folder>```

### Multi-GPU Training

In Multi-GPU environment, run training using ```mpirun``` with the number of GPUs.

```bash
mpirun -n <GPUS> python main.py ctdet --dataset <coco or pascal> --arch <res or dlav0> --num_layers <18 or 34> --batch_size <batch size> --gpus <gpus to use>
```
### Mixed Precision Training

To enable mixed precision, add the flags ```--mixed_precision``` and ```--channel_last```. 

```bash
mpirun -n <GPUS> python main.py ctdet --mixed_precision --channel_last --dataset <coco or pascal> --arch <res or dlav0> --num_layers <18 or 34> --batch_size <batch size> --gpus <gpus to use>
```

### Additional arguments:

Please see the help description by executing

```bash
python main.py -h
```

## Validation

You can use the ```test.py``` script for AP/mAP validation:

```bash
python test.py ctdet --dataset <coco or pascal> --data_dir <coco or pascal root folder> --arch <resnet or dlav0> --num_layers <number layers> --checkpoint <path to checkpoint file> --gpus <gpu to use>
```

You can also recalculate the AP .txt files for a series using ```test.py```:

```bash
python test.py ctdet --dataset <coco or pascal> --data_dir <coco or pascal root folder> --arch <resnet or dlav0> --num_layers <number layers> --checkpoint_dir <root folder of checkpoints> --gpus <gpu to use>
```

## Demo

The ```dataset```,  ```arch``` and  ```num_layers``` parameters must match with loaded weights.

```bash
python demo.py ctdet --dataset <coco or pascal> --arch <resnet or dlav0> --num_layers <number of layers> --checkpoint <path to *.h5 file> --demo <test_image.jpg> --gpus <gpu to use> --debug 1
```

Set the ```debug``` parameter controls the outputs from the detector:
 * 0 for no output
 * 1 for bounding boxes over original image
 * 2 for bounding boxes over original image, bounding boxes over rescaled input image and heatmaps per class

## Pretrained weights and benchmarks

The hyperparameters used for mixed precision and full precision training were the same as the ones used by the Pytorch repository.

Hyperparameter search was not made thoroughly so almost 1-2AP variance is expected. Doubling the number of epochs with large learning rate increase the AP by around 2-3 points.
 
The evaluation scripts from the official datasets is used to calculate AP/mAP. 

### PascalVOC

| Arch. | GPUs | MP*1 | LR | Epochs | LR steps | Batch size per GPU | mAP | Pretrained parameters (Click to download) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ResNet18 |  1 x V100 | No | 1.25e-4 | 70 | 45,60 | 32 | 68.40  | TODO
| ResNet18 |  1 x V100 | Yes | 1.25e-4 | 70 | 45,60 | 32 | 69.24  | TODO
| DLAv0 |  2 x V100 | No | 1.25e-4 | 70 | 45,60 | 16 | 74.72  | TODO
| DLAv0 |  1 x V100 | Yes | 1.25e-4 | 70 | 45,60 | 32 | 75.06 | TODO

### COCO


| Arch. | GPUs | MP*1 | LR | Epochs | LR steps | Batch size per GPU | AP (0.5-0.95 IoU)| Pretrained parameters (Click to download) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ResNet18 |  4 x V100 | No | 5.e-4 | 140 | 90,120 | 32 | 25.18 | TODO
| ResNet18 |  4 x V100 | Yes | 5.e-4 | 140 | 90,120 | 32 | 24.81 | TODO
| DLAv0 |  8 x V100 | No | 5e-4 | 140 | 90,120 | 16 | 31.77  | TODO
| DLAv0 |  4 x V100 | Yes | 5e-4 | 140 | 90,120 | 32 | 31.85  | TODO

Note: The AP might  

## Export nnp file

`src/save_nnp.py` provides the way to export nnp file. Specify the `--dataset`, `--arch` and `--num_layers` options and the output will be saved in `exp/ctdet/your_exp_id/`.

```bash
cd src/

# For example, to export DLAv034 nnp file, use the following command.
python save_nnp.py ctdet --dataset <coco or pascal> --arch dlav0 --num_layers 34
```

Please run `python save_nnp.py -h` to see which network is supported.

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
