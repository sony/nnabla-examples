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
import nnabla.functions as F
import nnabla.parametric_functions as PF
import numpy as np

from models.networks.initializers import torch_initializer
from models.networks.model_resnet import resnet_imagenet
from nnabla.initializer import ConstantInitializer, NormalInitializer
from nnabla.utils.save import save


def pf_deconvolution(x, ochannels, kernel, stride=(1, 1), pad=(1, 1), dilation=(2, 2), with_bias=False, w_init=None, b_init=None, channel_last=False):
    x = PF.deconvolution(
        x, ochannels, kernel,
        pad=pad,
        stride=stride,
        dilation=dilation,
        w_init=w_init,
        group=1,
        with_bias=with_bias,
        b_init=b_init,
        channel_last=channel_last)
    return x


def pf_convolution(x, ochannels, kernel, pad=(1, 1), stride=(1, 1), with_bias=False, w_init=None, b_init=None, channel_last=False):
    return PF.convolution(x, ochannels, kernel, stride=stride, pad=pad,
                          with_bias=with_bias, w_init=w_init, b_init=b_init, channel_last=channel_last)


class PoseResNet(object):
    def __init__(
            self,
            num_layers,
            heads,
            head_conv,
            training=True,
            channel_last=False,
            **kwargs):
        self.num_layers = num_layers
        self.training = training
        self.heads = heads
        self.head_conv = head_conv
        self.backbone_model = resnet_imagenet
        self.n_init = NormalInitializer(0.001)
        self.channel_last = channel_last
        # used for deconv num_layers
        self.ochannels = ([256, 256, 256])
        self.kernels_size = ([4, 4, 4])

    def __call__(self, x):
        if not isinstance(x, nn._variable.Variable):
            input_variable = nn.Variable(x.shape)
            if isinstance(x, np.ndarray):
                input_variable.d = x
            else:
                input_variable.data = x
        else:
            input_variable = x
        axes = 3 if self.channel_last else 1
        r, hidden = self.backbone_model(
            input_variable,
            num_classes=1000,
            num_layers=self.num_layers,
            shortcut_type='b',
            test=not self.training,
            channel_last=self.channel_last)
        with nn.parameter_scope("upsample1"):
            kernel_size = self.kernels_size[0]
            features = pf_deconvolution(
                hidden['r4'], self.ochannels[0], (kernel_size, kernel_size),
                pad=(1, 1),
                stride=(2, 2),
                dilation=(1, 1),
                w_init=self.n_init,
                with_bias=False,
                channel_last=self.channel_last
            )

            features = PF.batch_normalization(
                features,
                axes=[axes],
                batch_stat=self.training,
                param_init={'gamma': ConstantInitializer(
                    1), 'beta': ConstantInitializer(0)},
            )

            features = F.relu(features)

        with nn.parameter_scope("upsample2"):
            kernel_size = self.kernels_size[1]
            features = pf_deconvolution(
                features, self.ochannels[1], (kernel_size, kernel_size),
                pad=(1, 1),
                stride=(2, 2),
                dilation=(1, 1),
                w_init=self.n_init,
                with_bias=False,
                channel_last=self.channel_last
            )

            features = F.relu(PF.batch_normalization(
                features,
                axes=[axes],
                batch_stat=self.training,
                param_init={'gamma': ConstantInitializer(
                    1), 'beta': ConstantInitializer(0)}))

        with nn.parameter_scope("upsample3"):
            kernel_size = self.kernels_size[2]
            features = pf_deconvolution(
                features, self.ochannels[2], (kernel_size, kernel_size),
                pad=(1, 1),
                stride=(2, 2),
                dilation=(1, 1),
                w_init=self.n_init,
                with_bias=False,
                channel_last=self.channel_last
            )
            features = F.relu(PF.batch_normalization(
                features,
                axes=[axes],
                batch_stat=self.training,
                param_init={'gamma': ConstantInitializer(1), 'beta': ConstantInitializer(0)})
                )

        output = []
        for head in sorted(self.heads):
            num_output = self.heads[head]
            rng = np.random.RandomState(313)
            b_init_param = -2.19 if head == 'hm' else 0.0
            if self.head_conv > 0:
                with nn.parameter_scope(head + "_conv1"):
                    w_init_param = torch_initializer(
                        features.shape[axes], (3, 3)) if head == 'hm' else self.n_init
                    out = pf_convolution(features,
                                         self.head_conv,
                                         (3, 3),
                                         pad=(1, 1),
                                         stride=(1, 1),
                                         with_bias=True,
                                         w_init=w_init_param,
                                         b_init=ConstantInitializer(
                                             b_init_param),
                                         channel_last=self.channel_last,
                                         )
                    out = F.relu(out)
                with nn.parameter_scope(head + "_final"):
                    w_init_param = torch_initializer(
                        out.shape[axes], (1, 1)) if head == 'hm' else self.n_init
                    out = pf_convolution(
                        out, num_output, (1, 1),
                        pad=(0, 0),
                        stride=(1, 1),
                        with_bias=True,
                        w_init=w_init_param,
                        b_init=ConstantInitializer(b_init_param),
                        channel_last=self.channel_last
                    )

            else:
                with nn.parameter_scope(head + "_final"):
                    out = pf_convolution(
                        features,
                        num_output, (1, 1),
                        pad=(0, 0),
                        stride=(1, 1),
                        with_bias=True,
                        w_init=w_init_param,
                        b_init=ConstantInitializer(b_init_param),
                        channel_last=self.channel_last
                        )
            output.append(out)
        return output


def get_pose_net(num_layers, heads, head_conv, training, channel_last=False, opt=None, batch_size=None):
    model = PoseResNet(num_layers, heads, head_conv,
                       training=training, channel_last=channel_last)
    return model


def load_weights(pretrained_model_dir, num_layers, channel_last):
    layout = 'nhwc' if channel_last else 'nchw'
    nn.load_parameters(os.path.join(pretrained_model_dir,
                                    "resnet{}_{}_imagenet.h5".format(num_layers, layout)))
