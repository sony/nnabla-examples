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
import math
import glob
import re
import nnabla as nn
import numpy as np
import cv2


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def test_index_generation(skip, n_out, len_in):
    """
    params:
    skip: if skip even number;
    N_out: number of frames of the network;
    len_in: length of input frames

    example:
  len_in | N_out  | times | (no skip)                  |   (skip)
    5    |   3    |  4/2  | [0,1], [1,2], [2,3], [3,4] | [0,2],[2,4]
    7    |   3    |  5/3  | [0,1],[1,2][2,3]...[5,6]   | [0,2],[2,4],[4,6]
    5    |   5    |  2/1  | [0,1,2] [2,3,4]            | [0,2,4]
    """
    # number of input frames for the network
    n_in = 1 + n_out // 2
    # input length should be enough to generate the output frames
    assert n_in <= len_in

    sele_list = []
    if skip:
        right = n_out  # init
        while right <= len_in:
            h_list = [right - n_out + x for x in range(n_out)]
            l_list = h_list[::2]
            right += (n_out - 1)
            sele_list.append([l_list, h_list])
    else:
        right = n_out  # init
        right_in = n_in
        while right_in <= len_in:
            h_list = [right - n_out + x for x in range(n_out)]
            l_list = [right_in - n_in + x for x in range(n_in)]
            right += (n_out - 1)
            right_in += (n_in - 1)
            sele_list.append([l_list, h_list])
    # check if it covers the last image, if not, we should cover it
    if (skip) and (right < len_in - 1):
        h_list = [len_in - n_out + x for x in range(n_out)]
        l_list = h_list[::2]
        sele_list.append([l_list, h_list])
    if (not skip) and (right_in < len_in - 1):
        right = len_in * 2 - 1
        h_list = [right - n_out + x for x in range(n_out)]
        l_list = [len_in - n_in + x for x in range(n_in)]
        sele_list.append([l_list, h_list])
    return sele_list


def read_image(img_path):
    '''read one image from img_path
    Return img: HWC, BGR, [0,1], numpy
    '''
    img_gt = cv2.imread(img_path)
    img = img_gt.astype(np.float32) / 255.
    return img


def read_seq_imgs_(img_seq_path):
    '''read a sequence of images'''
    img_path_l = glob.glob(img_seq_path + '/*')
    # img_path_l.sort(key=lambda x: int(os.path.basename(x)[:-4]))
    img_path_l.sort(key=lambda x: int(
        re.search(r'\d+', os.path.basename(x)).group()))
    img_l = [read_image(v) for v in img_path_l]
    # stack to TCHW, RGB, [0,1]
    imgs = np.stack(img_l, axis=0)
    imgs = imgs[:, :, :, [2, 1, 0]]
    imgs = nn.Variable.from_numpy_array(
        np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2))))
    return imgs


def tensor2img(tensor, out_type=np.uint8):
    '''
    Converts a tensor into an image Numpy array
    Input: 3D(C,H,W), any range, RGB channel order
    Output: 3D(H,W,C), [0,255], np.uint8 (default)
    '''
    img_np = np.clip(tensor, 0, 1)  # clamp
    img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR

    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def bgr2ycbcr(img, only_y=True):
    '''bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


####################
# metric
####################

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) *
                                                            (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    if img1.ndim == 2:
        return ssim(img1, img2)

    if img1.ndim == 3:
        ssims = []
        for ch_idx in img1.shape[2]:
            ssims.append(ssim(img1[ch_idx], img2[ch_idx]))
        return np.array(ssims).mean()

    raise ValueError('Wrong input image dimensions.')
