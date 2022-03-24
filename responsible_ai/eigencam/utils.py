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
from PIL import Image

import base64
from io import BytesIO


from IPython.display import display, Javascript
from google.colab.output import eval_js


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


def resize_image_for_yolo(img_orig, w=608, h=608):

    im_h, im_w, _ = img_orig.shape

    if (w * 1.0 / im_w) < (h * 1. / im_h):
        new_w = w
        new_h = int((im_h * w) / im_w)
    else:
        new_h = h
        new_w = int((im_w * h) / im_h)

    patch = cv2.resize(img_orig, (new_w, new_h)) / 255.
    img = np.ones((h, w, 3), np.float32) * 0.5
    # resize
    x0 = int((w - new_w) / 2)
    y0 = int((h - new_h) / 2)
    img[y0:y0 + new_h, x0:x0 + new_w] = patch

    return img, new_w, new_h


def decode_img_str(img_str):
    decimg = base64.b64decode(img_str.split(',')[1], validate=True)
    decimg = Image.open(BytesIO(decimg))
    decimg = np.array(decimg, dtype=np.uint8)
    decimg = cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)
    return decimg


def encode_img(img):
    _, encimg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    img_str = encimg.tostring()
    img_str = "data:image/jpeg;base64," + \
        base64.b64encode(img_str).decode('utf-8')
    return img_str
