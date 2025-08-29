#!/bin/bash

curl -O https://zenodo.org/records/16962677/files/Resnet-50.nnp?download=1

python -m nnabla.utils.cli.cli convert -E0 -S4- -b1 Resnet-50.nnp html/resnet.onnx
