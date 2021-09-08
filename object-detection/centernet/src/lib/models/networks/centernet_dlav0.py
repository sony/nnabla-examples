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


"""
DLA primitives and full network models.
"""

import os
import nnabla as nn
import nnabla.parametric_functions as PF
import nnabla.functions as F
from nnabla.initializer import UniformInitializer, ConstantInitializer, NormalInitializer
from nnabla.logger import logger
from nnabla.utils.save import save
from .model_dlav0 import dla_imagenet, DLAUp
import numpy as np


def xavier_initializer(inmaps, outmaps, kernel):
    d = np.sqrt(6. / (np.prod(kernel) * inmaps + np.prod(kernel) * outmaps))
    return UniformInitializer((-d, d))


def torch_initializer(inmaps, kernel):
    d = np.sqrt(1. / (np.prod(kernel) * inmaps))
    return UniformInitializer((-d, d))


def pf_convolution(x, ochannels, kernel, pad=(1, 1), stride=(1, 1), with_bias=False, w_init=None, b_init=None, channel_last=False):
    return PF.convolution(x, ochannels, kernel, stride=stride, pad=pad,
                          with_bias=with_bias, w_init=w_init, b_init=b_init, channel_last=channel_last)


class PoseDLA(object):
    def __init__(
            self,
            num_layers,
            heads,
            head_conv,
            training=True,
            channel_last=False,
            **kwargs):

        self.n_init = NormalInitializer(0.001)
        self.backbone_model = DLAUp
        self.num_layerse = num_layers
        self.training = training
        self.heads = heads
        self.head_conv = head_conv
        self.channel_last = channel_last
        # TODO add DLA variations
        self.axes = 3 if self.channel_last else 1

    def __call__(self, x):
        if not isinstance(x, nn._variable.Variable):
            input_variable = nn.Variable(x.shape)
            if isinstance(x, np.ndarray):
                input_variable.d = x
            else:
                input_variable.data = x
        else:
            input_variable = x

        features = self.backbone_model(
            input_variable, test=not self.training, channel_last=self.channel_last)

        output = []
        for head in sorted(self.heads):
            num_output = self.heads[head]
            if self.head_conv > 0:
                with nn.parameter_scope(head + "_conv1"):
                    b_init_param = -2.19 if head == 'hm' else 0.0
                    w_init_param = torch_initializer(
                        features.shape[self.axes], (3, 3)) if head == 'hm' else self.n_init
                    out = pf_convolution(
                        features,
                        self.head_conv,
                        (3, 3),
                        pad=(1, 1),
                        stride=(1, 1),
                        w_init=w_init_param,
                        b_init=ConstantInitializer(b_init_param),
                        with_bias=True,
                        channel_last=self.channel_last
                    )
                    out = F.relu(out)
                with nn.parameter_scope(head + "_final"):
                    w_init_param = torch_initializer(
                        features.shape[self.axes], (1, 1)) if head == 'hm' else self.n_init
                    out = pf_convolution(
                        out,
                        num_output,
                        (1, 1),
                        pad=(0, 0),
                        stride=(1, 1),
                        w_init=w_init_param,
                        b_init=ConstantInitializer(b_init_param),
                        with_bias=True,
                        channel_last=self.channel_last
                    )
            else:
                with nn.parameter_scope(head + "_final"):
                    w_init_param = torch_initializer(
                        features.shape[self.axes], (1, 1)) if head == 'hm' else self.n_init
                    out = pf_convolution(
                        features,
                        num_output,
                        (1, 1),
                        pad=(0, 0),
                        stride=(1, 1),
                        w_init=w_init_param,
                        with_bias=True,
                        channel_last=self.channel_last
                    )
            output.append(out)
        return output


def get_pose_net(num_layers, heads, head_conv, training, channel_last=False, opt=None, batch_size=None):
    model = PoseDLA(num_layers, heads, head_conv,
                    training=training, channel_last=channel_last)
    return model


def load_weights(pretrained_model_dir, num_layers, channel_last):
    layout = 'nhwc' if channel_last else 'nchw'
    nn.load_parameters(os.path.join(pretrained_model_dir,
                                    "dla{}_{}_imagenet.h5".format(num_layers, layout)))
