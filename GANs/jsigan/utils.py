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

import os
import sys
import imageio
import numpy as np
import nnabla.functions as F

common_utils_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
sys.path.append(common_utils_path)

from neu.yaml_wrapper import read_yaml
from neu.misc import AttrDict
from neu.comm import CommunicatorWrapper
from neu.variable_utils import set_persistent_all


def get_hw_boundary(patch_boundary, h, w, pH, sH, pW, sW):
    """
    Calculate height and width of patch
    """
    h_low_ind = max(pH * sH - patch_boundary, 0)
    h_high_ind = min((pH + 1) * sH + patch_boundary, h)
    w_low_ind = max(pW * sW - patch_boundary, 0)
    w_high_ind = min((pW + 1) * sW + patch_boundary, w)

    return h_low_ind, h_high_ind, w_low_ind, w_high_ind


def depth_to_space(x, sf):
    """
    x: (B, H, W, sf*sf)
    reshape x to (B, H, W, sf,sf,1) then transpose to (B, H, sf, W, sf, 1) & finally reshape (B, H*sf, W*sf, 1)
    return: (B, H*sf, W*sf, 1)
    """
    x_sz = x.shape
    if x_sz[3] == sf*sf:
        x = F.reshape(x, (x_sz[0], x_sz[1], x_sz[2], sf, sf, 1))
        x = F.reshape(F.transpose(x, (0, 1, 3, 2, 4, 5)),
                      (x_sz[0], x_sz[1] * sf, x_sz[2] * sf, 1))
    else:
        ch = 64
        x = F.reshape(x, (x_sz[0], x_sz[1], x_sz[2], sf, sf, ch))
        x = F.reshape(F.transpose(x, (0, 1, 3, 2, 4, 5)),
                      (x_sz[0], x_sz[1] * sf, x_sz[2] * sf, ch))
    return x


def trim_patch_boundary(img, patch_boundary, h, w, pH, sH, pW, sW, sf):
    """
    Remove both rows and columns to reduce edge effect 
    around patch edges.
    """
    # trim rows
    if pH * sH >= patch_boundary:
        img = img[:, patch_boundary * sf:, :, :]
    if (pH + 1) * sH + patch_boundary <= h:
        img = img[:, :-patch_boundary * sf, :, :]

    # trim columns
    if pW * sW >= patch_boundary:
        img = img[:, :, patch_boundary * sf:, :]
    if (pW + 1) * sW + patch_boundary <= w:
        img = img[:, :, :-patch_boundary * sf, :]

    return img


def compute_psnr(img_orig, img_out, peak):
    """
    Calculate PSNR value
    """
    mse = np.mean((img_orig - img_out) ** 2)
    psnr = 10 * np.log10(1.*1. / mse)
    return psnr


def get_learning_rate(init_lr, iteration, lr_stair_decay_points, lr_decreasing_factor):
    """
    Calculate learning rate decay
    """
    epoch_lr_to_be_decayed_boundaries = [
        y * (iteration) for y in lr_stair_decay_points]
    epoch_lr_to_be_decayed_value = [init_lr * (lr_decreasing_factor ** y) for y in
                                    range(len(lr_stair_decay_points) + 1)]
    lr = init_lr
    if epoch_lr_to_be_decayed_boundaries[0] < iteration <= epoch_lr_to_be_decayed_boundaries[1]:
        lr = epoch_lr_to_be_decayed_value[1]
    if iteration > epoch_lr_to_be_decayed_boundaries[1]:
        lr = epoch_lr_to_be_decayed_value[2]

    return lr


def save_results_yuv(pred, index, test_img_dir):
    """
    Saves generated YUV image seperately by channel  
    """
    test_pred = np.squeeze(pred)
    test_pred = np.clip(test_pred, 0, 1) * 1023
    test_pred = np.uint16(test_pred)

    # split image
    pred_y = test_pred[:, :, 0]
    pred_u = test_pred[:, :, 1]
    pred_v = test_pred[:, :, 2]

    # save prediction - must be saved in separate channels due to 16-bit pixel depth
    imageio.imwrite(os.path.join(test_img_dir, "{}-y_pred.png".format(str(int(index) + 1).zfill(2))),
                    pred_y)
    imageio.imwrite(os.path.join(test_img_dir, "{}-u_pred.png".format(str(int(index) + 1).zfill(2))),
                    pred_u)
    imageio.imwrite(os.path.join(test_img_dir, "{}-v_pred.png".format(str(int(index) + 1).zfill(2))),
                    pred_v)
