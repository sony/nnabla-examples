
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

import numpy as np


def CustomFunc(inputs, func, *args, **kwargs):
    im_l = func(inputs[0], *args, **kwargs)
    im_ab = func(inputs[1], *args, **kwargs)
    warp_ba = func(inputs[2], *args, **kwargs)
    warp_aba = func(inputs[3], *args, **kwargs)
    # im_gbl_ab  = func(inputs[4], *args, **kwargs)
    # bgr_mc_im = func(inputs[5], *args, **kwargs)
    layer_data = [im_l, im_ab, warp_ba, warp_aba]
    # layer_data = [im_l, im_ab, warp_ba, warp_aba, im_gbl_ab, bgr_mc_im]
    for l in range(5):
        layer = inputs[4 + l]
        err_ba = func(layer[0], *args, **kwargs)
        err_ab = func(layer[1], *args, **kwargs)

        layer_data.append([err_ba, err_ab])

    return layer_data

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, inputs):
        for t in self.transforms:
            inputs = t(inputs)
        return inputs

class CenterPad(object):
    def __init__(self, image_size):
        self.height = image_size[0]
        self.width = image_size[1]

    def __call__(self, image):
        # pad the image to 16:9
        # pad height
        I = np.array(image)

        # for padded input
        height_old = np.size(I, 0)
        width_old = np.size(I, 1)
        old_size = [height_old, width_old]
        height = self.height
        width = self.width
        I_pad = np.zeros((height, width, np.size(I, 2)))

        ratio = height / width
        if height_old / width_old == ratio:
            if height_old == height:
                return Image.fromarray(I.astype(np.uint8))
            new_size = [int(x * height / height_old) for x in old_size]
            I_resize = resize(I, new_size, mode="reflect", preserve_range=True, clip=False, anti_aliasing=True)
            return Image.fromarray(I_resize.astype(np.uint8))

        if height_old / width_old > ratio:  # pad the width and crop
            new_size = [int(x * width / width_old) for x in old_size]
            I_resize = resize(I, new_size, mode="reflect", preserve_range=True, clip=False, anti_aliasing=True)
            width_resize = np.size(I_resize, 1)
            height_resize = np.size(I_resize, 0)
            start_height = (height_resize - height) // 2
            I_pad[:, :, :] = I_resize[start_height : (start_height + height), :, :]
        else:  # pad the height and crop
            new_size = [int(x * height / height_old) for x in old_size]
            I_resize = resize(I, new_size, mode="reflect", preserve_range=True, clip=False, anti_aliasing=True)
            width_resize = np.size(I_resize, 1)
            height_resize = np.size(I_resize, 0)
            start_width = (width_resize - width) // 2
            I_pad[:, :, :] = I_resize[:, start_width : (start_width + width), :]

        return Image.fromarray(I_pad.astype(np.uint8))
class RGB2Lab(object):
    def __call__(self, inputs):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """

    def __call__(self, inputs):
        image_lab = color.rgb2lab(inputs[0])
        warp_ba_lab = color.rgb2lab(inputs[2])
        warp_aba_lab = color.rgb2lab(inputs[3])
        # im_gbl_lab = color.rgb2lab(inputs[4])

        inputs[0] = image_lab[:, :, :1]  # l channel
        inputs[1] = image_lab[:, :, 1:]  # ab channel
        inputs[2] = warp_ba_lab  # lab channel
        inputs[3] = warp_aba_lab  # lab channel
        # inputs[4] = im_gbl_lab[:, :, 1:]    # ab channel

        return inputs

class RGB2Lab(object):
    def __call__(self, inputs):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """

    def __call__(self, inputs):
        image_lab = color.rgb2lab(inputs[0])
        warp_ba_lab = color.rgb2lab(inputs[2])
        warp_aba_lab = color.rgb2lab(inputs[3])
        # im_gbl_lab = color.rgb2lab(inputs[4])

        inputs[0] = image_lab[:, :, :1]  # l channel
        inputs[1] = image_lab[:, :, 1:]  # ab channel
        inputs[2] = warp_ba_lab  # lab channel
        inputs[3] = warp_aba_lab  # lab channel
        # inputs[4] = im_gbl_lab[:, :, 1:]    # ab channel

        return inputs

class CenterCrop(object):
    """Crop the given PIL Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = (h - th) // 2
        j = (w - tw) // 2
        return i, j, th, tw

    def __call__(self, inputs):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        if self.padding > 0:
            inputs = CustomFunc(inputs, pad, self.padding)

        if type(inputs) is list:
            i, j, h, w = self.get_params(inputs[0], self.size)
        else:
            i, j, h, w = self.get_params(inputs, self.size)
        return CustomFunc(inputs, F.crop, i, j, h, w)

def pad(img, padding, fill=0):
    """Pad the given PIL Image on all sides with the given "pad" value.
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
        raise TypeError("img should be PIL Image. Got {}".format(type(img)))

    if not isinstance(padding, (numbers.Number, tuple)):
        raise TypeError("Got inappropriate padding arg")
    if not isinstance(fill, (numbers.Number, str, tuple)):
        raise TypeError("Got inappropriate fill arg")

    if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
        raise ValueError(
            "Padding must be an int or a 2, or 4 element tuple, not a " + "{} element tuple".format(len(padding))
        )

    return ImageOps.expand(img, border=padding, fill=fill)
