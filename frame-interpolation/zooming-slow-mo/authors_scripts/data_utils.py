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

import math
import numpy as np


def cubic(x):
    """
    Cubic kernel
    """
    absx = np.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * (
        (absx <= 1).astype(type(absx))) + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((
            (absx > 1) * (absx <= 2)).astype(type(absx)))


def calculate_weights_indices(in_length, out_length, scale, kernel_width, antialiasing):
    """
    Get weights and indices
    """
    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = np.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = np.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = np.repeat(left.reshape(out_length, 1), P).reshape(out_length, P) +  \
        np.broadcast_to(np.linspace(
            0, P - 1, P).reshape(1, P), (out_length, P))

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = np.repeat(
        u.reshape(out_length, 1), P).reshape(out_length, P) - indices

    # apply cubic kernel
    if (scale < 1) and (antialiasing):
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = np.sum(weights, 1).reshape(out_length, 1)
    weights = weights / np.repeat(weights_sum, P).reshape(out_length, P)

    # If a column in weights is all zero, get rid of it. only consider the first and last column.
    weights_zero_tmp = np.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices[:, 1:P-1]
        weights = weights[:, 1:P-1]
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices[:, 0:P-1]
        weights = weights[:, 0:P-1]

    weights = np.ascontiguousarray(weights)
    indices = np.ascontiguousarray(indices)
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


def imresize_np(img, scale, antialiasing=True):
    """
    The scale should be the same for H and W
    Input: img: Numpy, HWC BGR [0,1]
    Output: HWC BGR [0,1] w/o round
    """
    in_h, in_w, in_channels = img.shape
    _, out_h, out_w = in_channels, math.ceil(
        in_h * scale), math.ceil(in_w * scale)
    kernel_width = 4

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_h, indices_h, sym_len_hs, sym_len_he = calculate_weights_indices(
        in_h, out_h, scale, kernel_width, antialiasing)
    weights_w, indices_w, sym_len_ws, sym_len_we = calculate_weights_indices(
        in_w, out_w, scale, kernel_width, antialiasing)

    # process h dimension
    # symmetric copying
    img_aug = np.ndarray((in_h + sym_len_hs + sym_len_he,
                          in_w, in_channels), dtype='float32')
    img_aug[sym_len_hs:sym_len_hs+in_h, :, :] = img
    sym_patch = img[:sym_len_hs, :, :]
    inv_idx = np.arange(sym_patch.shape[0] - 1, -1, -1).astype('int64')
    sym_patch_inv = np.take(sym_patch, inv_idx, 0)
    img_aug[0:sym_len_hs:, :, :] = sym_patch_inv
    sym_patch = img[-sym_len_he:, :, :]
    inv_idx = np.arange(sym_patch.shape[0] - 1, -1, -1).astype('int64')
    sym_patch_inv = np.take(sym_patch, inv_idx, 0)
    img_aug[sym_len_hs + in_h:sym_len_hs +
            in_h + sym_len_he, :, :] = sym_patch_inv
    out_1 = np.ndarray((out_h, in_w, in_channels), dtype='float32')
    kernel_width = weights_h.shape[1]

    for i in range(out_h):
        idx = int(indices_h[i][0])
        out_1[i, :, 0] = np.dot(
            img_aug[idx:idx + kernel_width, :, 0].transpose(1, 0), weights_h[i])
        out_1[i, :, 1] = np.dot(
            img_aug[idx:idx + kernel_width, :, 1].transpose(1, 0), weights_h[i])
        out_1[i, :, 2] = np.dot(
            img_aug[idx:idx + kernel_width, :, 2].transpose(1, 0), weights_h[i])

    # process w dimension
    # symmetric copying
    out_1_aug = np.ndarray(
        (out_h, in_w + sym_len_ws + sym_len_we, in_channels), dtype='float32')
    out_1_aug[:, sym_len_ws:sym_len_ws + in_w] = out_1
    sym_patch = out_1[:, :sym_len_ws, :]
    inv_idx = np.arange(sym_patch.shape[1] - 1, -1, -1).astype('int64')
    sym_patch_inv = np.take(sym_patch, inv_idx, 1)
    out_1_aug[:, 0:0+sym_len_ws] = sym_patch_inv
    sym_patch = out_1[:, -sym_len_we:, :]
    inv_idx = np.arange(sym_patch.shape[1] - 1, -1, -1).astype('int64')
    sym_patch_inv = np.take(sym_patch, inv_idx, 1)
    out_1_aug[:, sym_len_ws + in_w:sym_len_ws +
              in_w + sym_len_we] = sym_patch_inv
    out_2 = np.ndarray((out_h, out_w, in_channels), dtype='float32')
    kernel_width = weights_w.shape[1]

    for i in range(out_w):
        idx = int(indices_w[i][0])
        out_2[:, i, 0] = np.dot(
            out_1_aug[:, idx:idx + kernel_width, 0], weights_w[i])
        out_2[:, i, 1] = np.dot(
            out_1_aug[:, idx:idx + kernel_width, 1], weights_w[i])
        out_2[:, i, 2] = np.dot(
            out_1_aug[:, idx:idx + kernel_width, 2], weights_w[i])

    return out_2
