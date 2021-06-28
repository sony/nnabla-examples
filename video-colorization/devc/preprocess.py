# Copyright (c) 2021 Sony Corporation. All Rights Reserved.
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
import torch
import torchvision.utils as vutils

rgb_from_xyz = np.array(
    [[3.24048134, -0.96925495, 0.05564664], [-1.53715152, 1.87599, -0.20404134], [-0.49853633, 0.04155593, 1.05731107]],dtype = np.float32)
l_norm, ab_norm = 1.0, 1.0
l_mean, ab_mean = 50.0, 0

def rgb2lab(img):
    return color.rgb2lab(img)

def centerpad(img, newsize):
    height_old = img.shape[0]
    width_old = img.shape[1]
    height = newsize[0]
    width = newsize[1]
    img_pad = np.zeros((height,width,img.shape[2]))
    ratio = height/width
    old_size = [height_old, width_old]
    if height_old / width_old == ratio:
        if height_old == height:
            return img
        new_size = [int(x * height / height_old) for x in old_size]
        img_resize = resize(img, new_size, mode="reflect", preserve_range=True, clip=False, anti_aliasing=True)
        return img_resize
    if height_old / width_old > ratio:  # pad the width and crop
        new_size = [int(x * width / width_old) for x in old_size]
        img_resize= resize(img, new_size, mode="reflect", preserve_range=True, clip=False, anti_aliasing=True)
        height_resize = img_resize.shape[0]
        start_height = (height_resize - height) // 2
        img_pad[:, :, :] = img_resize[start_height : (start_height + height), :, :]
    else: 
        new_size = [int(x * height / height_old) for x in old_size]
        img_resize = resize(img, new_size, mode="reflect", preserve_range=True, clip=False, anti_aliasing=True)
        width_resize = img_resize.shape[1]
        start_width = (width_resize - width) // 2
        img_pad[:, :, :] = img_resize[:, start_width : (start_width + width), :]
    return img_pad


def pad_func(img, padding, fill=0):
    """
    Pad the given PIL Image on all sides with the given "pad" value.

    Args:
        img (PIL Image): Image to be padded.
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill: Pixel fill value. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.

    Returns:
        PIL Image: Padded image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if not isinstance(padding, (numbers.Number, tuple)):
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(fill, (numbers.Number, str, tuple)):
        raise TypeError('Got inappropriate fill arg')

    if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
        raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                         "{} element tuple".format(len(padding)))

    return ImageOps.expand(img, border=padding, fill=fill)

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
    #if not _is_pil_image(img):
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
        w,h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = (h - th) // 2
        j = (w - tw) // 2
        return i, j, th, tw
    inputs  = Image.fromarray(inputs.astype(np.uint8))
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))

    if padding > 0:
        inputs = custom_func(inputs, pad_func, padding)

    if type(inputs) is list:
        i, j, h, w = get_params(inputs[0], size)
    else:
        i, j, h, w = get_params(inputs, size)
    image = crop_func(inputs, i, j, h, w)
    return np.array(image)

def normalize_fn(arr, mean, std):
    """
    Normalize an image with mean and standard deviation.

    Args:
        arr (nd array): arr/image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.

    Returns:
        norm_arr: Normalized Tensor image.
    """
    if(arr.ndim < 3):
        arr = np.expand_dims(arr, axis = 0)        
    if arr.shape[0] == 1:
        norm_arr = arr.copy()        
        norm_arr[0, :, :] = (arr[0, :, :] - mean) / std
    else:        
        for i in range(arr.shape[0]):
            norm_arr = arr.copy()
            norm_arr[i, :, :] = (arr[i, :, :] - mean[i]) / std[i]
    return norm_arr

def normalize(inputs):
    inputs[0:1, :, :] = normalize_fn(inputs[0:1, :, :], 50, 1)
    inputs[1:3, :, :] = normalize_fn(inputs[1:3, :, :], (0, 0), (1, 1))
    return inputs

def lab2rgb(input):
    """
    shape : n * 3* h *w
    """
    input_trans = input.transpose(0,2,3,1)  # n * h * w * 3
    L, a, b = input_trans[:, :, :, 0:1], input_trans[:, :, :, 1:2], input_trans[:, :, :, 2:]
    y = (L + 16.0) / 116.0
    x = (a / 500.0) + y
    z = y - (b / 200.0)

    neg_mask = z < 0
    z[neg_mask] = 0
    xyz = np.concatenate((x,y,z), axis=3)
    mask = xyz > 0.2068966
    mask_xyz = xyz.copy()
    mask_xyz[mask] = np.power(xyz[mask], 3.0)
    mask_xyz[~mask] = (xyz[~mask] - 16.0 / 116.0) / 7.787
    mask_xyz[:, :, :, 0] = mask_xyz[:, :, :, 0] * 0.95047
    mask_xyz[:, :, :, 2] = mask_xyz[:, :, :, 2] * 1.08883
    rgb_trans = np.matmul(mask_xyz.reshape(-1,3),rgb_from_xyz).reshape(input.shape[0],input.shape[2],input.shape[3],3)
    rgb = np.transpose(rgb_trans,(0,3,1,2))
    mask = rgb > 0.0031308
    mask_rgb = rgb.copy()
    mask_rgb[mask] = 1.055 * np.power(rgb[mask], 1 / 2.4) - 0.055
    mask_rgb[~mask] = rgb[~mask] * 12.92

    neg_mask = mask_rgb < 0
    large_mask = mask_rgb > 1
    mask_rgb[neg_mask] = 0
    mask_rgb[large_mask] = 1
    return mask_rgb    

def uncenter_l(l):
    return l * l_norm + l_mean

def batch_lab2rgb_transpose(img_l_mc, img_ab_mc, nrow=8):
    img_l_mc = img_l_mc
    img_ab_mc = img_ab_mc

    assert img_l_mc.ndim == 4 and img_ab_mc.ndim == 4, "only for batch input"

    img_l = img_l_mc * l_norm + l_mean
    img_ab = img_ab_mc * ab_norm + ab_mean
    pred_lab = np.concatenate((img_l, img_ab), axis=1)
    pred_lab = torch.from_numpy(pred_lab)
    grid_lab = vutils.make_grid(pred_lab, nrow=nrow).numpy().astype("float64")
    return (np.clip(color.lab2rgb(grid_lab.transpose((1, 2, 0))), 0, 1) * 255).astype("uint8")

def save_frames(image, image_folder, index=None, frame_name=None):
    if image is not None:
        image = np.clip(image, 0, 255).astype(np.uint8)
        if frame_name:
            io.imsave(os.path.join(image_folder, frame_name), image)
        else:
            io.imsave(os.path.join(image_folder, str(index).zfill(5) + '.jpg'), image)

def frames2vid(frame_folder,frame_shape, output_dir, filename):
    frames = [img for img in os.listdir(frame_folder) if img.endswith(".jpg")]
    #sort the frames in order
    frames.sort()
    # get the height and width
    height, width = frame_shape
    print(f"writing to video file: {os.path.join(output_dir, filename)}")
    video = cv2.VideoWriter(os.path.join(output_dir, filename), cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), 24, (width, height))
    for frame in frames:
        video.write(cv2.imread(os.path.join(frame_folder, frame)))
    video.release()

