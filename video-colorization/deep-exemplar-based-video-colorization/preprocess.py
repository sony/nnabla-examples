
# Copyright 2021 Sony Group Corporation
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
import numbers
import collections
import cv2
import numpy as np
from skimage.transform import resize
from PIL import Image, ImageOps
import nnabla.functions as F
from skimage import color, io


def rgb2lab(img):
    return color.rgb2lab(img)


def centerpad(img, newsize):
    height_old = img.shape[0]
    width_old = img.shape[1]
    height = newsize[0]
    width = newsize[1]
    img_pad = np.zeros((height, width, img.shape[2]))
    ratio = height / width
    old_size = [height_old, width_old]
    if height_old / width_old == ratio:
        if height_old == height:
            return img
        new_size = [int(x * height / height_old) for x in old_size]
        img_resize = resize(
            img,
            new_size,
            mode="reflect",
            preserve_range=True,
            clip=False,
            anti_aliasing=True)
        return img_resize
    if height_old / width_old > ratio:  # pad the width and crop
        new_size = [int(x * width / width_old) for x in old_size]
        img_resize = resize(
            img,
            new_size,
            mode="reflect",
            preserve_range=True,
            clip=False,
            anti_aliasing=True)
        height_resize = img_resize.shape[0]
        start_height = (height_resize - height) // 2
        img_pad[:, :, :] = img_resize[start_height: (
            start_height + height), :, :]
    else:
        new_size = [int(x * height / height_old) for x in old_size]
        img_resize = resize(
            img,
            new_size,
            mode="reflect",
            preserve_range=True,
            clip=False,
            anti_aliasing=True)
        width_resize = img_resize.shape[1]
        start_width = (width_resize - width) // 2
        img_pad[:, :, :] = img_resize[:, start_width: (start_width + width), :]
    return img_pad


def crop_func(img, i, j, h, w):
    """Crop the given PIL Image.

    Args:
        img (PIL Image): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.

    Returns:
        PIL Image: Cropped image.
    """
    # if not _is_pil_image(img):
    #    raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.crop((j, i, j + w, i + h))


def center_crop(inputs, size, padding=0):
    """
    Crop the given PIL Image at a random location.

    Args:
        img (PIL Image): Image to be cropped.
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        Returns:
            PIL Image: Cropped image.
    """
    def get_params(img, output_size):
        """
        Get parameters for 'crop' for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to 'crop' for random crop.
        """
        #w, h = img.shape[:-1]
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = (h - th) // 2
        j = (w - tw) // 2
        return i, j, th, tw
    inputs = Image.fromarray(inputs.astype(np.uint8))
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))

    if isinstance(inputs, list):
        i, j, h, w = get_params(inputs[0], size)
    else:
        i, j, h, w = get_params(inputs, size)
    image = crop_func(inputs, i, j, h, w)
    return np.array(image)


def standardize(arr, mean, std):
    """
    Standardize the input with mean and standard deviation.

    Args:
        arr (nd array): arr/image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.

    Returns:
        norm_arr: Normalized Tensor image.
    """
    if (arr.ndim < 3):
        arr = np.expand_dims(arr, axis=0)
    if arr.shape[0] == 1:
        norm_arr = arr.copy()
        norm_arr[0, :, :] = (arr[0, :, :] - mean) / std
    else:
        for i in range(arr.shape[0]):
            norm_arr = arr.copy()
            norm_arr[i, :, :] = (arr[i, :, :] - mean[i]) / std[i]
    return norm_arr


def normalize(image):
    """
    Normalize the input

    Args:
        image :  Input image (numpy)
    Returns:
        Normalized image in dtype float32

    """
    image[0, :, :] = standardize(image[0:1, :, :], 50, 1)
    image[1:3, :, :] = standardize(image[1:3, :, :], (0, 0), (1, 1))
    return image.astype(np.float32)


def lab2rgb(input):
    input_trans = F.split(input, axis=1)
    L, a, b = F.split(input, axis=1)
    y = (L + 16.0) / 116.0
    x = (a / 500.0) + y
    z = y - (b / 200.0)
    neg_mask = F.less_scalar(z, 0).apply(need_grad=False)
    z = z * F.logical_not(neg_mask)
    mask_Y = F.greater_scalar(y, 0.2068966).apply(need_grad=False)
    mask_X = F.greater_scalar(x, 0.2068966).apply(need_grad=False)
    mask_Z = F.greater_scalar(z, 0.2068966).apply(need_grad=False)
    Y_1 = (y ** 3) * mask_Y
    Y_2 = L / (116. * 7.787) * F.logical_not(mask_Y)
    var_Y = Y_1 + Y_2

    X_1 = (x ** 3) * mask_X
    X_2 = (x - 16. / 116.) / 7.787 * F.logical_not(mask_X)
    var_X = X_1 + X_2

    Z_1 = (z ** 3) * mask_Z
    Z_2 = (z - 16. / 116.) / 7.787 * F.logical_not(mask_Z)
    var_Z = Z_1 + Z_2

    X = 0.95047 * var_X
    Y = 1.00000 * var_Y
    Z = 1.08883 * var_Z

    var_R = X * 3.2406 + Y * -1.5372 + Z * -0.4986
    var_G = X * -0.9689 + Y * 1.8758 + Z * 0.0415
    var_B = X * 0.0557 + Y * -0.2040 + Z * 1.0570

    mask_R = F.greater_scalar(var_R, 0.0031308).apply(need_grad=False)
    n_mask_R = F.logical_not(mask_R)
    R_1 = (1.055 * (F.maximum2(var_R, n_mask_R) ** (1 / 2.4)) - 0.055) * mask_R
    R_2 = (12.92 * var_R) * n_mask_R
    var_R = R_1 + R_2

    mask_G = F.greater_scalar(var_G, 0.0031308).apply(need_grad=False)
    n_mask_G = F.logical_not(mask_G)
    G_1 = (1.055 * (F.maximum2(var_G, n_mask_G) ** (1 / 2.4)) - 0.055) * mask_G
    G_2 = (12.92 * var_G) * n_mask_G
    var_G = G_1 + G_2

    mask_B = F.greater_scalar(var_B, 0.0031308).apply(need_grad=False)
    n_mask_B = F.logical_not(mask_B)
    B_1 = (1.055 * (F.maximum2(var_B, n_mask_B) ** (1 / 2.4)) - 0.055) * mask_B
    B_2 = (12.92 * var_B) * n_mask_B
    var_B = B_1 + B_2
    return F.stack(var_R, var_G, var_B, axis=1)


def uncenter_l(l, conf):
    return l * conf.l_norm + conf.l_mean


def batch_lab2rgb_transpose(conf, img_l_mc, img_ab_mc, nrow=8):
    img_l_mc = img_l_mc
    img_ab_mc = img_ab_mc

    assert img_l_mc.ndim == 4 and img_ab_mc.ndim == 4, "only for batch input"

    img_l = img_l_mc * conf.l_norm + conf.l_mean
    img_ab = img_ab_mc * conf.ab_norm + conf.ab_mean
    pred_lab = np.concatenate((img_l, img_ab), axis=1)
    grid_lab = pred_lab.squeeze().astype(np.float64)
    return (
        np.clip(
            color.lab2rgb(
                grid_lab.transpose(
                    (1, 2, 0))), 0, 1) * 255).astype("uint8")


def save_frames(image, image_folder, index=None, frame_name=None):
    if image is not None:
        image = np.clip(image, 0, 255).astype(np.uint8)
        if frame_name:
            io.imsave(os.path.join(image_folder, frame_name), image)
        else:
            io.imsave(
                os.path.join(
                    image_folder,
                    str(index).zfill(5) +
                    '.jpg'),
                image)


def frames2vid(frame_folder, frame_shape, output_dir, filename):
    frames = sorted([img for img in os.listdir(
        frame_folder) if img.endswith(".jpg")])
    # get the height and width
    height, width = frame_shape
    print(f"writing to video file: {os.path.join(output_dir, filename)}")
    video = cv2.VideoWriter(
        os.path.join(
            output_dir, filename), cv2.VideoWriter_fourcc(
            'D', 'I', 'V', 'X'), 24, (width, height))
    for frame in frames:
        video.write(cv2.imread(os.path.join(frame_folder, frame)))
    video.release()
