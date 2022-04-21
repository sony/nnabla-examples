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

import numpy as np

import nnabla as nn
import nnabla.functions as F
from nnabla.ext_utils import get_extension_context

import nnabla_cliport.parametric_functions as CPF
from nnabla_cliport.models.model import Model


class CLIPortAttention(Model):
    def __init__(self, scope_name, image_encoder, text_encoder, outdim=1, training=False, half_precision=None):
        super().__init__(scope_name)
        self._image_encoder = image_encoder
        self._text_encoder = text_encoder
        self._outdim = outdim
        self._training = training
        self._half_precision = half_precision

    def compute_features(self, rgbd, text, apply_softmax=True):
        # rgbd shape: 6xHxW (default will be 6x320x160)
        # BCHW
        # input is padded in original code
        rgbd_h = rgbd.shape[-2]
        rgbd_w = rgbd.shape[-1]
        max_dim = np.max(rgbd.shape[-2:])
        pads = (max_dim - np.array(rgbd.shape[-2:])) / 2
        pad_width = (pads[0], pads[0], pads[1], pads[1])
        rgbd = F.pad(rgbd, pad_width=pad_width)
        transporter_rgbd = self._preprocess(rgbd, dist='transporter')
        transporter_rgbd.need_grad = False
        clip_rgbd = self._preprocess(rgbd, dist='clip')
        clip_rgbd.need_grad = False
        clip_rgb = clip_rgbd[:, 0:3, :, :]
        clip_rgb.need_grad = False
        encoded_image, image_features = self._encode_image(
            clip_rgb, with_attn_pool=False)
        encoded_text = self._encode_text(text)
        # BCHW H -> -2, W -> -1
        output_size = (clip_rgb.shape[-2], clip_rgb.shape[-1])
        with nn.parameter_scope(self.scope_name):
            spatial_features, mid_features = self._spatial(
                transporter_rgbd, output_size)
            semantic_features = self._semantic(
                mid_features, encoded_image, image_features, encoded_text, output_size)
            logits = CPF.add_fusion(spatial_features, semantic_features)
            logits = logits[:, :, pads[0]:pads[0] +
                            rgbd_h, pads[1]:pads[1]+rgbd_w]

            logits = F.transpose(logits, axes=(0, 2, 3, 1))  # [B, H, W, R]
            output = logits.reshape(
                (logits.shape[0], np.prod(logits.shape[1:])))
            if apply_softmax:
                output = F.softmax(output, axis=len(output.shape) - 1)
                output = output.reshape(logits.shape)
        return output

    def _spatial(self, rgbd, output_size):
        with nn.parameter_scope('stream_one'):
            with nn.parameter_scope('spatial'):
                return CPF.cliport_spatial(rgbd, output_size, self._outdim)

    def _semantic(self, spatial, encoded_image, image_features, encoded_text, output_size):
        with nn.parameter_scope('stream_two'):
            with nn.parameter_scope('semantic'):
                return CPF.cliport_semantic(spatial, encoded_image, image_features, encoded_text, output_size, self._outdim, self._training)

    def _encode_image(self, rgb, with_attn_pool):
        if self._half_precision is not None:
            half_precision = get_extension_context(
                "cudnn", type_config="half", device_id=self._half_precision)
            with nn.context_scope(half_precision):
                x, image_features = self._image_encoder.encode_image(
                    rgb, with_attn_pool)
        else:
            x, image_features = self._image_encoder.encode_image(
                rgb, with_attn_pool)
        # use the prepool layer's output as encoded image
        x.need_grad = False
        for feature in image_features:
            feature.need_grad = False
        return x, image_features

    def _encode_text(self, tokens):
        if not isinstance(tokens, nn.Variable):
            tokens_var = nn.Variable((tokens.shape))
            tokens_var.d = tokens
        else:
            tokens_var = tokens
        if self._half_precision is not None:
            half_precision = get_extension_context(
                "cudnn", type_config="half", device_id=self._half_precision)
            with nn.context_scope(half_precision):
                encoded_text = self._text_encoder.encode_text(tokens_var)
        else:
            encoded_text = self._text_encoder.encode_text(tokens_var)
        encoded_text.need_grad = False
        return encoded_text

    def _preprocess(self, image, dist):
        """Pre-process input (subtract mean, divide by std)."""

        transporter_color_mean = [0.18877631, 0.18877631, 0.18877631]
        transporter_color_std = [0.07276466, 0.07276466, 0.07276466]
        transporter_depth_mean = 0.00509261
        transporter_depth_std = 0.00903967

        franka_color_mean = [0.622291933, 0.628313992, 0.623031488]
        franka_color_std = [0.168154213, 0.17626014, 0.184527364]
        franka_depth_mean = 0.872146842
        franka_depth_std = 0.195743116

        clip_color_mean = [0.48145466, 0.4578275, 0.40821073]
        clip_color_std = [0.26862954, 0.26130258, 0.27577711]

        # choose distribution
        if dist == 'clip':
            color_mean = clip_color_mean
            color_std = clip_color_std
        elif dist == 'franka':
            color_mean = franka_color_mean
            color_std = franka_color_std
        else:
            color_mean = transporter_color_mean
            color_std = transporter_color_std

        if dist == 'franka':
            depth_mean = franka_depth_mean
            depth_std = franka_depth_std
        else:
            depth_mean = transporter_depth_mean
            depth_std = transporter_depth_std

        # normalize
        def cast_shape(stat, image_shape):
            variable = nn.Variable.from_numpy_array(np.array(stat))
            variable = F.reshape(variable, shape=(1, *variable.shape, 1, 1))
            variable = F.broadcast(variable, shape=(
                image_shape[0], variable.shape[1], image_shape[2], image_shape[3]))
            variable.persistent = True
            return variable

        color_mean = cast_shape(color_mean, image.shape)
        color_std = cast_shape(color_std, image.shape)
        depth_mean = cast_shape(depth_mean, image.shape)
        depth_std = cast_shape(depth_std, image.shape)

        rgb_part = (image[:, :3, :, :] / 255 - color_mean) / color_std
        depth_part = (image[:, 3:, :, :] - depth_mean) / depth_std

        return F.concatenate(rgb_part, depth_part, axis=1)
