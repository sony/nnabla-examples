# Copyright 2021 Sony Corporation.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import nnabla as nn

from .networks.centernet_resnet import get_pose_net as get_centernet_resnet
from .networks.centernet_dlav0 import get_pose_net as get_centernet_dlav0
from .networks.centernet_resnet import load_weights as load_weights_resnet
from .networks.centernet_dlav0 import load_weights as load_weights_dlav0

_model_factory = {
    'resnet': get_centernet_resnet,  # default Resnet with deconv
    'dlav0': get_centernet_dlav0
}

_pretraind_weights_factory = {
    'resnet': load_weights_resnet,
    'dlav0': load_weights_dlav0,
}


def create_model(arch, heads, head_conv, num_layers, training, channel_last, pretrained_model_dir=None, **kwargs):
    get_model = _model_factory[arch]
    model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv,
                      training=training, channel_last=channel_last, **kwargs)
    if pretrained_model_dir:
        load_weights = _pretraind_weights_factory[arch]
        load_weights(pretrained_model_dir, num_layers, channel_last)
    return model


def load_model(model, path, clear=False):
    if clear:
        nn.clear_parameters()
    print("Loading weights from {}".format(path))
    nn.load_parameters(path)


def load_nnp_model(path, batch_size, output_num):
    from nnabla.utils.nnp_graph import NnpLoader

    nnp = NnpLoader(path)
    network_names = nnp.get_network_names()
    assert (len(network_names) > 0)
    graph = nnp.get_network(network_names[0], batch_size=batch_size)
    inputs = list(graph.inputs.keys())[0]
    outputs = list(graph.outputs.keys())
    x = graph.inputs[inputs]

    output_list = list()
    for i in range(output_num):
        output_list.append(graph.outputs[outputs[i]])

    return (x, output_list)
