# Developer Readme

This readme contains info  for extending and debugging CenterNet source code.

## Folder structure 


```
src
|---- scripts
|---- lib
    |----datasets
        |---- dataset
        |---- external
    |---- detectors
    |---- external
    |---- models
        |---- networks
    |---- opts.py
    |---- trains
    |---- tools
    |---- voc_eval_lib (external)
        |---- datasets
        |---- model
        |---- utils

```

The program entry points (main.py, test.py and demo.py) are located at ```src``` folder. Program scripts are mainly located at ```lib``` folder.

## Adding new Network

New networks should be placed at  ```models\networks```, registered to the factory class at ```models\model.py``` and added as an option at ```arch``` argument at ```opts.py```.

CenterNet uses an encoder-decoder esque architecture, so all models currently implemented are split between backbone (```model_dlav0.py, model_resnet.py```) and full model (```centernet_dlav0.py, centernet_resnet.py```). 

Please download pretrained weights

## Training Phase

Training definition is defined at ```main.py``` file, which instantiates all necessary objects (data iterators, loss functions, solver, comms, etc).

Training codes for object detection are located at ```lib/trains/ctdet.py```. This class basically gathers all components instantiated at ```main.py``` and run the training.


### Training Data

Training Data I/O and augmentation scripts are located at ```datasets\dataset``` folder. COCO annotation and pycocotools are adopted by default.

In case of PASCAL, please use the ```tools\get_pascal_voc.sh``` script to download the dataset with the ```.json``` converted annotations

The augmentation strategy is the same as CenterNet pytorch code, using OpenCV to apply the changes to the image. These are located at the correspondent ```dataset\xxx_config.py``` files.

The data iterator outputs ```self._variables = ('img', 'hm', 'hm_mask', 'wh_map', 'reg_map')```.

To add new datasets, one should add the correspondent ```xxx_config.py``` at ```datasets\dataset``` folder and register at ```datasets\dataset_factory.py``` factory class as well.

The ```ctdet_iterator_tester.py``` script can be used to debug data loading and test augmentation strategies.

### Loss Functions
 
 Loss functions are located at ```models/losses.py```. Currently L2 Loss, Focal Loss (implemented as ```_neg_loss```) and L1 Loss are implemented. 
 
 One difference between NNabla and Pytorch version is that L1 Loss is calculated between heatmaps straight for NNabla while Pytorch picks the top ```128``` scores, fetch the coordinates and apply L1 loss to it. 
 
 The models itself does not include ```sigmoid``` and ```clip``` operations on their architecture. These are added at ```CtdetLoss``` call and also at the detector separately.
 
### Detectors 

This class is mainly used for inference and mAP calculations. The function ```ctdet_decode``` reconstructs the outputs of the model into bounding boxes.

Sigmoid and a 3x3 max pooling is applied to the heatmap as post-processing operations. The latter is used to removed adjacent peaks analog to NMS.


## Inference/Demo

For testing, two scripts are available. ```demo.py``` runs inference given one image. This script is mainly used for empirical testing and general verification. ```test.py``` calculates validation error and AP/mAP over trained model.
