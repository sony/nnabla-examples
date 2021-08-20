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
This file contains a modified script written by Aliaksandr Siarohin.
The original code can be found in https://github.com/AliaksandrSiarohin/first-order-model/blob/master/logger.py
which is provided under CC BY-NC 4.0.
"""

import nnabla as nn
import nnabla.functions as F
import numpy as np
import matplotlib.pyplot as plt

from skimage.draw import circle


class Visualizer(object):
    def __init__(self, kp_size=5, draw_border=False, colormap='gist_rainbow'):
        self.kp_size = kp_size
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)

    def draw_image_with_kp(self, image, kp_array):
        image = np.copy(image)
        spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
        kp_array = spatial_size * (kp_array + 1) / 2
        num_kp = kp_array.shape[0]
        for kp_ind, kp in enumerate(kp_array):
            rr, cc = circle(kp[1], kp[0], self.kp_size, shape=image.shape[:2])
            image[rr, cc] = np.array(self.colormap(kp_ind / num_kp))[:3]
        return image

    def create_image_column_with_kp(self, images, kp):
        image_array = np.array([self.draw_image_with_kp(v, k)
                                for v, k in zip(images, kp)])
        return self.create_image_column(image_array)

    def create_image_column(self, images):
        if self.draw_border:
            images = np.copy(images)
            images[:, :, [0, -1]] = (1, 1, 1)
            images[:, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(images), axis=0)

    def create_image_grid(self, *args):
        out = []
        for arg in args:
            if type(arg) == tuple:
                out.append(self.create_image_column_with_kp(arg[0], arg[1]))
            else:
                out.append(self.create_image_column(arg))
        return np.concatenate(out, axis=1)

    def visualize(self, driving, source, out):
        images = []

        # Source image with keypoints
        if isinstance(source, nn.Variable):
            source = source.d
        kp_source = out['kp_source']['value'].d
        source = np.transpose(source, [0, 2, 3, 1])
        images.append((source, kp_source))

        # Equivariance visualization, not used when animation (eval)
        if 'transformed_frame' in out:
            transformed = out['transformed_frame'].d
            transformed = np.transpose(transformed, [0, 2, 3, 1])
            transformed_kp = out['transformed_kp']['value'].d
            images.append((transformed, transformed_kp))

        # Driving image with keypoints
        kp_driving = out['kp_driving']['value'].d
        if isinstance(driving, nn.Variable):
            driving = driving.d
        driving = np.transpose(driving, [0, 2, 3, 1])
        images.append((driving, kp_driving))

        # Deformed image
        if 'deformed' in out:
            deformed = out['deformed'].d
            deformed = np.transpose(deformed, [0, 2, 3, 1])
            images.append(deformed)

        # Result with and without keypoints
        prediction = out['prediction'].d
        prediction = np.transpose(prediction, [0, 2, 3, 1])
        if 'kp_norm' in out:
            kp_norm = out['kp_norm']['value'].d
            images.append((prediction, kp_norm))
        images.append(prediction)

        # Occlusion map
        if 'occlusion_map' in out:
            with nn.auto_forward():
                occlusion_map = F.tile(out['occlusion_map'], (1, 3, 1, 1))
                occlusion_map = F.interpolate(
                    occlusion_map, output_size=source.shape[1:3], mode='nearest')
            occlusion_map = np.transpose(occlusion_map.d, [0, 2, 3, 1])
            images.append(occlusion_map)

        # Deformed images according to each individual transform
        if 'sparse_deformed' in out:
            full_mask = []
            for i in range(out['sparse_deformed'].shape[1]):
                with nn.auto_forward():
                    image = out['sparse_deformed'][:, i]
                    image = F.interpolate(
                        image, output_size=source.shape[1:3], mode='nearest')
                    mask = F.tile(out['mask'][:, i:(i + 1)], (1, 3, 1, 1))
                    mask = F.interpolate(
                        mask, output_size=source.shape[1:3], mode='nearest')
                image = np.transpose(image.d, (0, 2, 3, 1))
                mask = np.transpose(mask.d, (0, 2, 3, 1))

                if i != 0:
                    color = np.array(self.colormap(
                        (i - 1) / (out['sparse_deformed'].shape[1] - 1)))[:3]
                else:
                    color = np.array((0, 0, 0))

                color = color.reshape((1, 1, 1, 3))

                images.append(image)
                if i != 0:
                    images.append(mask * color)
                else:
                    images.append(mask)

                full_mask.append(mask * color)

            images.append(sum(full_mask))

        image = self.create_image_grid(*images)
        image = (255 * image).astype(np.uint8)
        return image
