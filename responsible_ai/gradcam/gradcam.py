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

import cv2
import numpy as np


def gradcam(middle_layer):
    """
    Calculate GradCAM.

    Parameters
    ----------
    middle_layer: nn.Variable
        The layer of interest to apply GradCAM


    Returns
    ----------
    heatmap: ndarray
        2D array of same size as width and height of middle_layer
    """
    conv_layer_output = middle_layer.d
    conv_layer_grad = middle_layer.g
    pooled_grad = conv_layer_grad.mean(axis=(0, 2, 3), keepdims=True)
    heatmap = pooled_grad * conv_layer_output
    heatmap = np.maximum(heatmap, 0)    # ReLU
    heatmap = heatmap.mean(axis=(0, 1))
    max_v, min_v = np.max(heatmap), np.min(heatmap)
    if max_v != min_v:
        heatmap = (heatmap - min_v) / (max_v - min_v)
    return heatmap


def overlay_images(base_img, overlay_img, overlay_coef=1.0):
    """
    Overlay two images.

    Parameters
    ----------
    base_img: ndarray
        2D array
    overlay_img: ndarray
        2D array
    overlay_coef: float
        mix rate of overlay_img to base_img (overlay_img: overlay_coef, base_img: 1). 

    Returns
    ----------
    ret_img: ndarray
        2D array of overlaid image
    """
    # resize
    _overlay_img = cv2.resize(
        overlay_img, (base_img.shape[1], base_img.shape[0]))
    # normalize
    _overlay_img = 255 * _overlay_img / _overlay_img.max()
    _overlay_img = _overlay_img.astype('uint8')
    # color adjust
    _overlay_img = cv2.applyColorMap(_overlay_img, cv2.COLORMAP_JET)
    base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
    # overlay
    ret_img = _overlay_img * overlay_coef + base_img
    ret_img = 255 * ret_img / ret_img.max()
    ret_img = ret_img.astype('uint8')
    ret_img = cv2.cvtColor(ret_img, cv2.COLOR_BGR2RGB)
    return ret_img
