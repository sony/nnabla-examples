# Copyright 2022 Sony Group Corporation.
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

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

import re

import nnabla_cliport.parametric_functions as CPF
from nnabla_cliport.models.model import Model
from nnabla_cliport.clip.model import modified_resnet_no_pool, modified_resnet


class CLIPImageEncoder(Model):
    def __init__(self, scope_name, vision_layers=[3, 4, 6, 3], vision_width=64, embed_dim=1024, training=False):
        super().__init__(scope_name)
        self._vision_layers = vision_layers
        self._vision_width = vision_width
        self._embed_dim = embed_dim
        self._training = training

    def encode_image(self, image: nn.Variable, with_attn_pool=True):
        if len(image.shape) == 3:
            image = F.reshape(image, (1, *image.shape))
        mid_features = []
        inplanes = self._vision_width
        with nn.parameter_scope(self.scope_name):
            x = self._stem(image, self._vision_width)
            # prepool
            for i, blocks in enumerate(self._vision_layers):
                with nn.parameter_scope(f'layer{i+1}'):
                    stride = 1 if i == 0 else 2
                    x, inplanes = self._resnet_layer(
                        x, blocks, self._vision_width * (2**i), inplanes, stride=stride)
                    mid_features.append(x)

            # attnpool
            if with_attn_pool:
                output_dim = self._embed_dim
                embed_dim = self._vision_width * 32  # ResNet feature dimension
                image_resolution = image.shape[-1]
                vision_heads = self._vision_width * 32 // 64
                with nn.parameter_scope('attnpool'):
                    x = CPF.attention_pool_2d(
                        x, image_resolution // 32, embed_dim, vision_heads, output_dim)

        for mid_feature in mid_features:
            mid_feature.need_grad = False
        x.need_grad = False
        return x, mid_features

    def _stem(self, x, width):
        with nn.parameter_scope('conv1'):
            x = PF.convolution(x, outmaps=width // 2, kernel=(3, 3),
                               stride=(2, 2), pad=(1, 1), with_bias=False)
        with nn.parameter_scope('bn1'):
            x = PF.batch_normalization(x, batch_stat=self._training)
            x = F.relu(x)
        with nn.parameter_scope('conv2'):
            x = PF.convolution(x, outmaps=width // 2,
                               kernel=(3, 3), with_bias=False, pad=(1, 1))
        with nn.parameter_scope('bn2'):
            x = PF.batch_normalization(x, batch_stat=self._training)
            x = F.relu(x)
        with nn.parameter_scope('conv3'):
            x = PF.convolution(x, outmaps=width, kernel=(
                3, 3), with_bias=False, pad=(1, 1))
        with nn.parameter_scope('bn3'):
            x = PF.batch_normalization(x, batch_stat=self._training)
            x = F.relu(x)
        return F.average_pooling(x, kernel=(2, 2))

    def _resnet_layer(self, x, blocks, planes, inplanes, stride=1, expansion=4):
        for i in range(0, blocks):
            if i != 0:
                inplanes = planes * expansion
                stride = 1
            with nn.parameter_scope(f'{i}'):
                x = CPF.bottleneck(x, inplanes, planes, stride,
                                   expansion, training=self._training)
        return x, inplanes


class CLIPImageEncoderOld(object):
    def __init__(self):
        pass

    def encode_image(self, image, with_attn_pool=True):
        image = F.reshape(image, (1, *image.shape))
        if not isinstance(image, nn.Variable):
            image = nn.Variable.from_numpy_array(image)

        image_resolution = image.shape[-1]
        param_dict = nn.get_parameters()

        embed_dim = param_dict['text_projection'].shape[1]

        if 'visual/proj' not in param_dict:
            # Use ModifiedResNet
            vision_width = param_dict['visual/conv1/W'].shape[0] * 2
            print(f'vision width: {vision_width}')
            # vision_patch_size = param_dict['visual/conv1/W'].shape[-1]
            vision_layer_names = [k for k in param_dict.keys() if k.startswith('visual/layer')
                                  and k.endswith('/conv1/W')]
            vision_layers = {}
            for vision_layer_name in vision_layer_names:
                match = re.search(
                    r'layer([0-9]*).([0-9]*).*', vision_layer_name)
                layer_number = match.group(1)
                block_number = int(match.group(2)) + 1
                if layer_number in vision_layers:
                    if vision_layers[layer_number] < block_number:
                        vision_layers[layer_number] = block_number
                else:
                    vision_layers[layer_number] = block_number
            vision_layers = vision_layers.values()
            print(f'vision layers: {vision_layers}')
            vision_heads = vision_width * 32 // 64
            if with_attn_pool:
                encoded, mid_features = modified_resnet(image,
                                                        output_dim=embed_dim,
                                                        input_resolution=image_resolution,
                                                        heads=vision_heads,
                                                        layers=vision_layers,
                                                        width=vision_width)
            else:
                encoded, mid_features = modified_resnet_no_pool(image,
                                                                layers=vision_layers,
                                                                width=vision_width)
            encoded.need_grad = False  # Prevent updating parameters
            for mid_feature in mid_features:
                mid_feature.need_grad = False
            return encoded, mid_features
        else:
            raise NotImplementedError
