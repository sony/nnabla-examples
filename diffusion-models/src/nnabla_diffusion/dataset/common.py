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
import os
import queue
from typing import List

import numpy as np
from neu.comm import CommunicatorWrapper
from neu.datasets import _get_sliced_data_source
from nnabla import logger
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource
# set pillow as a backend so that we can use box sampling
from nnabla.utils.image_utils import backend_manager, imread, imresize
from nnabla_diffusion.config import DatasetConfig

backend_manager.set_backend("PilBackend")


def _get_img_hw(img, channel_first):
    return img.shape[-2:] if channel_first else img.shape[:-1]


def _resize_img(img: np.ndarray,
                size: int,
                channel_first=True,
                fix_aspect_ratio=True):
    # following guided-diffusion, apply box sampling multiple times -> bicubic

    # apply box sampling loop while the size of the shorter side is bigger than 2*size
    while True:
        h, w = _get_img_hw(img, channel_first)

        if min(h, w) < 2 * size:
            break

        img = imresize(img, size=(w // 2, h // 2), interpolate="box")

    # apply bicubic sampling
    if fix_aspect_ratio:
        scale = size / min(h, w)
        oh = max(size, int(np.ceil(h * scale)))
        ow = max(size, int(np.ceil(w * scale)))
    else:
        oh = size
        ow = size

    img = imresize(img, size=(ow, oh), interpolate="bicubic")

    return img


def resize_center_crop(img, size, channel_first=True):
    assert isinstance(size, int)
    assert len(img.shape) == 3

    rsz = _resize_img(img, size, channel_first, fix_aspect_ratio=True)

    h2, w2 = _get_img_hw(rsz, channel_first)
    h_off = (h2 - size) // 2
    w_off = (w2 - size) // 2
    rsz = rsz[:, h_off:h_off+size, w_off:w_off +
              size] if channel_first else rsz[h_off:h_off+size, w_off:w_off+size]

    h3, w3 = _get_img_hw(rsz, channel_first)
    assert h3 == size and w3 == size

    return rsz


def resize_random_crop(img, size, channel_first=True, max_crop_scale=1.25):
    assert isinstance(size, int)
    assert len(img.shape) == 3

    assert max_crop_scale >= 1.0
    pre_crop_size = np.random.randint(
        size, math.ceil(size * max_crop_scale) + 1)

    rsz = _resize_img(img, pre_crop_size,
                      channel_first=channel_first, fix_aspect_ratio=True)

    h2, w2 = _get_img_hw(rsz, channel_first)
    h_off = np.random.randint(h2 - size + 1)
    w_off = np.random.randint(w2 - size + 1)
    rsz = rsz[:, h_off:h_off+size, w_off:w_off +
              size] if channel_first else rsz[h_off:h_off+size, w_off:w_off+size]

    h3, w3 = _get_img_hw(rsz, channel_first)
    assert h3 == size and w3 == size

    return rsz


class SimpleDatasource(DataSource):
    def __init__(self,
                 conf: DatasetConfig,
                 img_paths: List[str],
                 *,
                 labels=None,
                 rng=None):
        super(SimpleDatasource, self).__init__(shuffle=conf.shuffle_dataset,
                                               rng=rng)

        if labels is not None:
            assert len(labels) == len(
                img_paths), "Number of images and labels must be the same."

        self.img_paths = img_paths
        self.labels = labels

        # check resolution
        assert len(conf.image_size) == 2, \
            "the length of image_size must be 2 (height, width)"

        self.im_size = conf.image_size
        self._variables = ["image", "label"]
        self._size = len(self.img_paths)
        self.on_memory = conf.on_memory
        self.fix_aspect_ratio = conf.fix_aspect_ratio
        self.random_crop = conf.random_crop
        self.channel_last = conf.channel_last

        if self.on_memory:
            self.images = [None for _ in range(self.size)]

        # keep data paths
        self.data_history = queue.Queue(maxsize=128)

        self.reset()

    def __len__(self):
        return self._size

    def get_last_data_path(self, size=1):
        ret = ()
        for i in range(size):
            ret += (self.data_history.get(), )

        return ret

    @property
    def size(self):
        return self._size

    def reset(self):
        self._indexes = self._rng.permutation(
            self.size) if self.shuffle else np.arange(self.size)
        super(SimpleDatasource, self).reset()

    def _get_data(self, i):
        image_idx = self._indexes[i]
        label = 0 if self.labels is None else self.labels[image_idx]

        # keep data paths
        if self.data_history.full():
            self.data_history.get()
        self.data_history.put(self.img_paths[image_idx])

        if self.on_memory and self.images[image_idx] is not None:
            return (self.images[image_idx], label)

        img = imread(self.img_paths[image_idx],
                     channel_first=not self.channel_last, num_channels=3)

        if self.fix_aspect_ratio:
            # perform resize and crop to keep original aspect ratio.

            if self.random_crop:
                # crop randomly so that cropped size equals to self.im_size
                img = resize_random_crop(
                    img, self.im_size[0], channel_first=not self.channel_last)
            else:
                # always crop the center region of an image
                img = resize_center_crop(
                    img, self.im_size[0], channel_first=not self.channel_last)
        else:
            # Breaking original aspect ratio, forcely resize image to self.im_size.
            img = _resize_img(
                img, self.im_size[0], channel_first=not self.channel_last, fix_aspect_ratio=False)

        # rescale pixel intensity to [-1, 1]
        img = img / 127.5 - 1

        if self.on_memory:
            self.images[image_idx] = img

        return (img, label)


SUPPORT_IMG_EXTS = [".jpg", ".png"]


def SimpleDataIterator(conf: DatasetConfig,
                       comm: CommunicatorWrapper = None,
                       label_creator_callback=None,
                       rng=None):
    # get all files
    paths = [os.path.join(conf.dataset_root_dir, x)
             for x in os.listdir(conf.dataset_root_dir) if os.path.splitext(x)[-1] in SUPPORT_IMG_EXTS]

    labels = None
    if label_creator_callback is not None:
        labels = [label_creator_callback(path) for path in paths]

    if len(paths) == 0:
        raise ValueError(f"[SimpleDataIterator] No data is found in {conf.dataset_root_dir}'. "
                         "Please make sure that you specify the correct directory path.")

    ds = SimpleDatasource(conf,
                          img_paths=paths,
                          labels=labels,
                          rng=rng)

    logger.info(f"Initialized data iterator. {ds.size} images are found.")

    ds = _get_sliced_data_source(ds, comm, conf.shuffle_dataset)

    return data_iterator(ds,
                         conf.batch_size,
                         with_memory_cache=False,
                         use_thread=True,
                         with_file_cache=False)
