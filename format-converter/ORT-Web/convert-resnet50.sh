#!/bin/bash

curl -O https://nnabla.org/pretrained-models/nnp_models/imagenet/Resnet-50/Resnet-50.nnp

python -m nnabla.utils.cli.cli convert -E0 -S4- -b1 Resnet-50.nnp html/resnet.onnx
