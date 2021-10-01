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
import argparse
import h5py
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
        #indices = indices.narrow(1, 1, P - 2)
        indices = indices[:, 1:P-1]
        #weights = weights.narrow(1, 1, P - 2)
        weights = weights[:, 1:P-1]
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        #indices = indices.narrow(1, 0, P - 2)
        indices = indices[:, 0:P-1]
        #weights = weights.narrow(1, 0, P - 2)
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
    in_H, in_W, in_C = img.shape
    _, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = np.ndarray((in_H + sym_len_Hs + sym_len_He,
                          in_W, in_C), dtype='float32')
    #img = img_aug[sym_len_Hs:sym_len_Hs+in_H,:,:]
    img_aug[sym_len_Hs:sym_len_Hs+in_H, :, :] = img

    sym_patch = img[:sym_len_Hs, :, :]
    inv_idx = np.arange(sym_patch.shape[0] - 1, -1, -1).astype('int64')
    sym_patch_inv = np.take(sym_patch, inv_idx, 0)
    #sym_patch_inv = img_aug[0:sym_len_Hs:,:,:]
    img_aug[0:sym_len_Hs:, :, :] = sym_patch_inv

    sym_patch = img[-sym_len_He:, :, :]
    inv_idx = np.arange(sym_patch.shape[0] - 1, -1, -1).astype('int64')
    sym_patch_inv = np.take(sym_patch, inv_idx, 0)
    #sym_patch_inv = img_aug[sym_len_Hs + in_H:sym_len_Hs + in_H + sym_len_He,:,:]
    img_aug[sym_len_Hs + in_H:sym_len_Hs +
            in_H + sym_len_He, :, :] = sym_patch_inv

    out_1 = np.ndarray((out_H, in_W, in_C), dtype='float32')
    kernel_width = weights_H.shape[1]

    for i in range(out_H):
        idx = int(indices_H[i][0])
        out_1[i, :, 0] = np.dot(
            img_aug[idx:idx + kernel_width, :, 0].transpose(1, 0), weights_H[i])
        out_1[i, :, 1] = np.dot(
            img_aug[idx:idx + kernel_width, :, 1].transpose(1, 0), weights_H[i])
        out_1[i, :, 2] = np.dot(
            img_aug[idx:idx + kernel_width, :, 2].transpose(1, 0), weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = np.ndarray(
        (out_H, in_W + sym_len_Ws + sym_len_We, in_C), dtype='float32')
    #out_1 = out_1_aug[:,sym_len_Ws:sym_len_Ws + in_W]
    out_1_aug[:, sym_len_Ws:sym_len_Ws + in_W] = out_1

    sym_patch = out_1[:, :sym_len_Ws, :]
    inv_idx = np.arange(sym_patch.shape[1] - 1, -1, -1).astype('int64')
    sym_patch_inv = np.take(sym_patch, inv_idx, 1)
    #sym_patch_inv = out_1_aug[:,0:0+sym_len_Ws]
    out_1_aug[:, 0:0+sym_len_Ws] = sym_patch_inv

    sym_patch = out_1[:, -sym_len_We:, :]
    inv_idx = np.arange(sym_patch.shape[1] - 1, -1, -1).astype('int64')
    sym_patch_inv = np.take(sym_patch, inv_idx, 1)
    #sym_patch_inv = out_1_aug[:,sym_len_Ws + in_W:sym_len_Ws + in_W + sym_len_We]
    out_1_aug[:, sym_len_Ws + in_W:sym_len_Ws +
              in_W + sym_len_We] = sym_patch_inv

    out_2 = np.ndarray((out_H, out_W, in_C), dtype='float32')
    kernel_width = weights_W.shape[1]
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[:, i, 0] = np.dot(
            out_1_aug[:, idx:idx + kernel_width, 0], weights_W[i])
        out_2[:, i, 1] = np.dot(
            out_1_aug[:, idx:idx + kernel_width, 1], weights_W[i])
        out_2[:, i, 2] = np.dot(
            out_1_aug[:, idx:idx + kernel_width, 2], weights_W[i])

    return out_2


# Command line arguments
data_parser = argparse.ArgumentParser(description='dataset path')

data_parser.add_argument('--hr-hdr-path', type=str, default="HDR_youtube_80.mat",
                         help='add path where HR-HDR dataset resides')

data_parser.add_argument('--lr-hdr-path', type=str, default="LR_HDR_youtube_80.mat",
                         help='where the LR-HDR images will be saved')

data_parser.add_argument('--reduction-factor', type=int, default=2,
                         help='by how much factor you want to reduce the size of the images')

data_parser.add_argument('--create-test-dataset', action='store_true', default=False,
                         help='If True, HDR_YUV is name of group in hdf5 dataset')

args = data_parser.parse_args()

if __name__ == "__main__":
    f_hr_hdr = h5py.File((args.hr_hdr_path))
    if args.create_test_dataset:
        imgs = f_hr_hdr['HDR_YUV']
    else:
        imgs = f_hr_hdr['HDR_data']
    img_lr_hdr_data = []

    for idx in range(len(imgs)):
        print("Image-", idx)
        img = imgs[idx]
        img_s = np.swapaxes(img, 0, 2)
        # Create LR images
        img_lr_hdr = imresize_np(img_s, 1 / args.reduction_factor, True)
        img_lr_hdr = np.swapaxes(img_lr_hdr, 0, 2)
        img_lr_hdr_data.append(img_lr_hdr)

    img_lr_hdr_data = np.array(img_lr_hdr_data)

    if args.create_test_dataset:
        with h5py.File(args.lr_hdr_path, "w") as f_lr_hdr:
            dset = f_lr_hdr.create_dataset('HDR_YUV', data=img_lr_hdr_data)
    else:
        with h5py.File(args.lr_hdr_path, "w") as f_lr_hdr:
            dset = f_lr_hdr.create_dataset('HDR_data', data=img_lr_hdr_data)

    f_hr_hdr.close()
