from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import nnabla as nn

from .networks.centernet_resnet import get_pose_net as get_centernet_resnet
from .networks.centernet_dlav0 import get_pose_net as get_centernet_dlav0
_model_factory = {
    'resnet': get_centernet_resnet,  # default Resnet with deconv
    'dlav0': get_centernet_dlav0
}


def create_model(arch, heads, head_conv, num_layers, training, channel_last, **kwargs):
    get_model = _model_factory[arch]
    model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv, training=training, channel_last=channel_last, **kwargs)
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
