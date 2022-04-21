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
from nnabla.ext_utils import get_extension_context

import numpy as np

from nnabla_cliport.models.model import Model
import nnabla_cliport.parametric_functions as CPF


class CLIPortTransport(Model):
    def __init__(self, scope_name, key_image_encoder, query_image_encoder, text_encoder,
                 outdim=3, kernel_dim=3, crop_size=64, rotations=36, training=False, half_precision=None):
        super().__init__(scope_name)
        self._key_image_encoder = key_image_encoder
        self._query_image_encoder = query_image_encoder
        self._text_encoder = text_encoder
        self._outdim = outdim
        self._kernel_dim = kernel_dim
        self._crop_size = crop_size
        self._pad_size = self._crop_size // 2
        self._rotations = rotations
        self._image_size = None
        self._training = training
        self._half_precision = half_precision

    def compute_features(self, rgbd, text, apply_softmax=True):
        # Perform cropping
        # x is (B, C, H, W)
        batch_size = rgbd.shape[0]
        rgbd = F.pad(rgbd, pad_width=(self._pad_size,
                     self._pad_size, self._pad_size, self._pad_size))
        crop = F.broadcast(F.reshape(rgbd, shape=(batch_size, 1, *rgbd.shape[1:])),
                           shape=(batch_size, self._rotations, *rgbd.shape[1:]))
        self._image_size = rgbd.shape[-2], rgbd.shape[-1]
        if not hasattr(self, '_rotation_matrices'):
            self._rotation_matrices = [nn.Variable(
                shape=(batch_size, 2, 3)) for _ in range(self._rotations)]
        crop = self._rotate(crop, self._rotation_matrices)
        crop = F.stack(*crop, axis=1)  # (B, R, C, H, W)
        if not hasattr(self, '_crop_y_indices'):
            self._crop_y_indices = nn.Variable(
                shape=(batch_size, self._rotations, self._crop_size))
        if not hasattr(self, '_crop_x_indices'):
            self._crop_x_indices = nn.Variable(
                shape=(batch_size, self._rotations, self._crop_size))
        (B, R, C, H, W) = crop.shape
        crop = F.reshape(crop, shape=(B * R, C, H, W))
        crop_region = (F.reshape(self._crop_y_indices, shape=(B * R, self._crop_size)),
                       F.reshape(self._crop_x_indices, shape=(B * R, self._crop_size)))
        crop = self._crop(crop, crop_region)

        logits = self._key(rgbd, text)
        kernel = self._query(crop, text)

        if B != 1:
            logits = F.split(logits, axis=0)
            kernel = F.reshape(kernel, shape=(B, R, *kernel.shape[1:]))
            kernels = F.split(kernel, axis=0)
            outputs = []
            for logit, kernel in zip(logits, kernels):
                logit = F.reshape(logit, shape=(1, *logit.shape))
                outputs.append(self._correlate(
                    logit, kernel, apply_softmax=apply_softmax))
            return F.stack(*outputs, axis=0)
        else:
            output = self._correlate(
                logits, kernel, apply_softmax=apply_softmax)
            return F.reshape(output, shape=(B, *output.shape))

    def set_pivot(self, pivot):
        batch_size = pivot.shape[0]
        if not hasattr(self, '_rotation_matrices'):
            self._rotation_matrices = [nn.Variable(
                shape=(batch_size, 2, 3)) for _ in range(self._rotations)]
        pivot = pivot + self._pad_size
        self._fill_rotation_matrices(
            self._image_size, self._rotation_matrices, pivot)
        if not hasattr(self, '_crop_y_indices'):
            self._crop_y_indices = nn.Variable(
                shape=(batch_size, self._rotations, self._crop_size))
        if not hasattr(self, '_crop_x_indices'):
            self._crop_x_indices = nn.Variable(
                shape=(batch_size, self._rotations, self._crop_size))
        crop_y_indices = np.asarray(
            [np.arange(pivot[i, 0]-self._pad_size, pivot[i, 0]+self._pad_size) for i in range(batch_size)])
        crop_y_indices = crop_y_indices.reshape(
            (batch_size, 1, *crop_y_indices.shape[1:]))
        self._crop_y_indices.d = np.broadcast_to(
            crop_y_indices, shape=self._crop_y_indices.shape)
        crop_x_indices = np.asarray(
            [np.arange(pivot[i, 1]-self._pad_size, pivot[i, 1]+self._pad_size) for i in range(batch_size)])
        crop_x_indices = crop_x_indices.reshape(
            (batch_size, 1, *crop_x_indices.shape[1:]))
        self._crop_x_indices.d = np.broadcast_to(
            crop_x_indices, shape=self._crop_x_indices.shape)

    def _key(self, rgbd, text):
        # rgbd shape: 6xHxW (default will be 6x320x320)
        # rgb shape: 3xHxW (default will be 3x320x320.)
        transporter_rgbd = self._preprocess(rgbd, dist='transporter')
        transporter_rgbd.need_grad = False
        clip_rgbd = self._preprocess(rgbd, dist='clip')
        clip_rgb = clip_rgbd[:, 0:3, :, :]
        clip_rgbd.need_grad = False
        clip_rgb.need_grad = False
        output_size = (clip_rgbd.shape[-2], clip_rgbd.shape[-1])
        encoded_image, image_features = self._encode_key_image(
            clip_rgb, with_attn_pool=False)
        encoded_text = self._encode_text(text)

        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope('key'):
                spatial_features, mid_features = spatial_features, mid_features = \
                    self._spatial(transporter_rgbd, output_size, self._outdim)
                semantic_features = self._semantic(mid_features, encoded_image,
                                                   image_features, encoded_text, output_size, self._outdim)
                with nn.parameter_scope('fusion'):
                    return CPF.conv_fusion(spatial_features, semantic_features, outmaps=3)

    def _query(self, rgbd, text):
        transporter_rgbd = self._preprocess(rgbd, dist='transporter')
        transporter_rgbd.need_grad = False
        clip_rgbd = self._preprocess(rgbd, dist='clip')
        clip_rgb = clip_rgbd[:, 0:3, :, :]
        clip_rgbd.need_grad = False
        clip_rgb.need_grad = False
        output_size = (clip_rgbd.shape[-2], clip_rgbd.shape[-1])
        # rgbd shape: 6xHxW (default will be 6x320x320)
        # rgb shape: 3xHxW (default will be 3x320x320.)
        encoded_image, image_features = self._encode_query_image(
            clip_rgb, with_attn_pool=False)
        encoded_text = self._encode_text(text)

        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope('query'):
                spatial_features, mid_features = spatial_features, mid_features = \
                    self._spatial(transporter_rgbd,
                                  output_size, self._kernel_dim)
                semantic_features = self._semantic(mid_features, encoded_image, image_features,
                                                   encoded_text, output_size, self._kernel_dim)
                with nn.parameter_scope('fusion'):
                    return CPF.conv_fusion(spatial_features, semantic_features, outmaps=3)

    def _spatial(self, rgbd, output_size, outdim):
        with nn.parameter_scope('stream_one'):
            with nn.parameter_scope('spatial'):
                return CPF.cliport_spatial(rgbd, output_size, outdim)

    def _semantic(self, spatial, encoded_image, image_features, encoded_text, output_size, outdim):
        with nn.parameter_scope('stream_two'):
            with nn.parameter_scope('semantic'):
                return CPF.cliport_semantic(spatial, encoded_image, image_features, encoded_text, output_size, outdim, self._training)

    def _encode_key_image(self, rgb, with_attn_pool):
        if self._half_precision is not None:
            half_precision = get_extension_context(
                "cudnn", type_config="half", device_id=self._half_precision)
            with nn.context_scope(half_precision):
                _, image_features = self._key_image_encoder.encode_image(
                    rgb, with_attn_pool)
        else:
            _, image_features = self._key_image_encoder.encode_image(
                rgb, with_attn_pool)

        # use the prepool layer's output as encoded image
        for feature in image_features:
            feature.need_grad = False
        return image_features[-1], image_features

    def _encode_query_image(self, rgb, with_attn_pool):
        if self._half_precision is not None:
            half_precision = get_extension_context(
                "cudnn", type_config="half", device_id=self._half_precision)
            with nn.context_scope(half_precision):
                _, image_features = self._query_image_encoder.encode_image(
                    rgb, with_attn_pool)
        else:
            _, image_features = self._query_image_encoder.encode_image(
                rgb, with_attn_pool)
        # use the prepool layer's output as encoded image
        for feature in image_features:
            feature.need_grad = False
        return image_features[-1], image_features

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

    def _correlate(self, logits, kernel, apply_softmax):
        x = F.convolution(logits, kernel, pad=(self._pad_size, self._pad_size))
        x = F.interpolate(x, output_size=(
            logits.shape[-2], logits.shape[-1]), half_pixel=True)
        x = x[:, :, self._pad_size:-self._pad_size,
              self._pad_size:-self._pad_size]

        output_shape = x.shape
        output = x.reshape((1, np.prod(output_shape)))
        if apply_softmax:
            output = F.softmax(output, axis=len(output.shape) - 1)
            output = output.reshape(output_shape[1:])
        else:
            output = output.reshape(output.shape[1:])
        return output

    def _fill_rotation_matrices(self, image_size, rotation_matrices, pivot):
        rotations = len(rotation_matrices)
        angles = [i * 2 * 180 / rotations for i in range(rotations)]
        for matrix, angle in zip(rotation_matrices, angles):
            padding = self._rotation_padding(image_size)
            padded_image_size = image_size[0] + \
                padding[0] * 2, image_size[1] + padding[2] * 2
            batch_size = matrix.shape[0]
            rotation_matrix = np.empty((batch_size, 2, 3))
            for i in range(batch_size):
                padded_pivot = (pivot[i, 0] + padding[0],
                                pivot[i, 1] + padding[2])
                rotation_matrix[i] = self._build_rotation_matrix(angle=angle,
                                                                 pivot=padded_pivot,
                                                                 scale=1,
                                                                 image_size=padded_image_size)
            matrix.d = rotation_matrix

    def _rotate(self, images, rotation_matrices):
        # images: (B, R, C, H, W)
        image_list = F.split(images, axis=1)
        rotated = []
        for i, (image, rotation_matrix) in enumerate(zip(image_list, rotation_matrices)):
            image_size = image.shape[-2], image.shape[-1]
            padding = self._rotation_padding(image_size)
            image = self._preprocess_rotation(image, padding)
            warped = self._warp_affine(image, rotation_matrix)
            warped = self._postprocess_rotation(warped, padding)
            rotated.append(warped)
        return rotated

    def _rotation_padding(self, image_size):
        (h, w) = image_size
        if h > w:
            size_diff = h - w
            pad_w_left = size_diff // 2
            pad_w_right = size_diff - pad_w_left
            padding = (0, 0, pad_w_left, pad_w_right)
        else:
            size_diff = w - h
            pad_h_up = size_diff // 2
            pad_h_bottom = size_diff - pad_h_up
            padding = (pad_h_up, pad_h_bottom, 0, 0)
        return padding

    def _preprocess_rotation(self, image, padding):
        return F.pad(image, pad_width=padding)

    def _postprocess_rotation(self, image, padding):
        h = image.shape[-2] - padding[0] - padding[1]
        w = image.shape[-1] - padding[2] - padding[3]
        return image[:, :, padding[0]:padding[0]+h, padding[2]:padding[2]+w]

    def _warp_affine(self, image, rotation_matrix):
        if image.shape[0] != rotation_matrix.shape[0]:
            raise RuntimeError(
                f'batch size of image and rotation_matrix mismatch! {image.shape[0]} != {rotation_matrix.shape[0]}')
        grid = F.affine_grid(rotation_matrix, size=(
            image.shape[-2], image.shape[-1]))
        return F.warp_by_grid(image, grid, channel_last=False)

    def _build_rotation_matrix(self, angle, pivot, scale, image_size):
        '''
        Compute rotation matrix for nnabla affine_grid.

        Args:
            angle (float): Rotation angle in degrees
            pivot (Tuple[int, int]): Center of the rotation.
            scale (float): Image scaling factor. The image will be scaled by this factor
            image_size (Tuple[int, int]): Size of the target image. (HxW)

        Returns:
            np.ndarray: Rotation matrix
        '''
        # nnabla rotates the image to the opposite direction of opencv
        angle = -angle
        (h, w) = image_size
        pivot = (2 * pivot[0] / (h - 1) - 1.0, 2 * pivot[1] / (w - 1) - 1.0)
        alpha = (1.0 / scale) * np.cos(angle/180.0 * np.pi)
        beta = (1.0 / scale) * np.sin(angle/180.0 * np.pi)
        cx = pivot[1]
        cy = pivot[0]
        return np.asarray([[alpha, beta, (1 - alpha) * cx - beta * cy],
                           [-beta, alpha, beta * cx + (1 - alpha) * cy]])

    def _crop(self, image, crop_region):
        """Crop image according to the given batch index
        Args:
            image (nn.Variable): the input image.
                Shape is (batch_size, num_channels) + image_shape.
            crop_region (Tuple[nn.Variable, ...]):
                tuple of indices variable for each axis.
                Each indices' shape is (batch_size, crop_size).
        Returns:
            nn.Variable: indexed_variable.
                Shape (batch_size, num_channels, crop_size1, crop_size2, ...).
        """
        batch_size, num_channels, *image_shape = image.shape
        crop_size = tuple(region.shape[1] for region in crop_region)
        total_crop_dim = np.prod(crop_size)
        if len(crop_region) != len(image_shape):
            raise ValueError(
                "the length of crop_region should be the same as the length of image_shape")
        for region in crop_region:
            if region.shape[0] != batch_size:
                raise ValueError(
                    "the batch-size of each region should be the same as the batch-size of image")
        # indices about batch axis
        batch_arange_array = np.tile(
            np.expand_dims(np.arange(batch_size), axis=[1, 2]),
            (1, num_channels, total_crop_dim)
        )
        batch_arange_array = np.reshape(
            batch_arange_array,
            (1, batch_size, num_channels, *crop_size)
        )
        batch_arange_array = batch_arange_array.astype(np.int32)
        batch_arange_indices = nn.Variable.from_numpy_array(batch_arange_array)
        # indices about channel axis
        channel_arange_array = np.tile(
            np.expand_dims(np.arange(num_channels), axis=[0, 2]),
            (batch_size, 1, total_crop_dim)
        )
        channel_arange_array = np.reshape(
            channel_arange_array,
            (1, batch_size, num_channels, *crop_size)
        )
        channel_arange_array = channel_arange_array.astype(np.int32)
        channel_arange_indices = nn.Variable.from_numpy_array(
            channel_arange_array)
        gather_indices = [batch_arange_indices, channel_arange_indices]
        common_shape = (1, batch_size, 1)
        common_tile_shape = (1, 1, num_channels)
        for axis_index, indices in enumerate(crop_region):
            dim = indices.shape[1]
            new_shape = common_shape + \
                tuple(dim if i == axis_index else 1 for i in range(len(crop_region)))
            new_tile_shape = common_tile_shape + \
                tuple(1 if i == axis_index else s for i,
                      s in enumerate(crop_size))
            indices = F.reshape(indices, new_shape)
            indices = F.tile(indices, new_tile_shape)
            gather_indices.append(indices)
        # indices
        gather_indices_var = F.concatenate(*gather_indices, axis=0)
        return F.gather_nd(image, gather_indices_var)

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
