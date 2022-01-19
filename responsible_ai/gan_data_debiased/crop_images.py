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

import os
import imageio
import numpy as np
from PIL import Image


def crop_images(input_path, output_path):
    """
    crop the aligned & cropped 178 * 218 images to 128 * 128
    Args:
        input_path (str) : path of image directory
        output_path (str) : path of cropped images to be saved
    """
    img_list = [f for f in os.listdir(
        input_path) if os.path.splitext(f)[1] == ".jpg"]
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    # to crop the aligned & cropped 178 * 218 images to 128 * 128
    cx = 121  # x-axis
    cy = 89  # y-axis
    c_pixels = 64  # align and center the images
    for _, item in enumerate(img_list):
        path = os.path.join(input_path, item)
        with open(path, 'rb') as image_path:
            img = Image.open(image_path)
            img = np.array(img.convert('RGB'))
            img = img[cx - c_pixels:cx + c_pixels, cy - c_pixels:cy + c_pixels]
            path = os.path.join(output_path, item)
            imageio.imwrite(path, img)


crop_images(r"\data\celeba\img_align_celeba", r"data\celeba\img_align_celeba")
