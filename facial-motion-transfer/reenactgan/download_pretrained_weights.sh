#!/bin/bash
# Copyright 2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

mkdir -p pretrained_weights/encoder
wget -P pretrained_weights/encoder https://nnabla.org/pretrained-models/nnabla-examples/GANs/reenactgan/encoder/weights.h5
wget -P pretrained_weights/encoder https://nnabla.org/pretrained-models/nnabla-examples/GANs/reenactgan/encoder/training_info.yaml

mkdir -p pretrained_weights/transformer/Kathleen2Donald_Trump
wget -P pretrained_weights/transformer/Kathleen2Donald_Trump https://nnabla.org/pretrained-models/nnabla-examples/GANs/reenactgan/transformer/weights.h5
wget -P pretrained_weights/transformer/Kathleen2Donald_Trump https://nnabla.org/pretrained-models/nnabla-examples/GANs/reenactgan/transformer/training_info.yaml

mkdir -p pretrained_weights/decoder/Donald_Trump
wget -P pretrained_weights/decoder/Donald_Trump https://nnabla.org/pretrained-models/nnabla-examples/GANs/reenactgan/decoder/weights.h5
wget -P pretrained_weights/decoder/Donald_Trump https://nnabla.org/pretrained-models/nnabla-examples/GANs/reenactgan/decoder/training_info.yaml
