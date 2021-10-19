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
import queue

import numpy as np
from nnabla import logger
from nnabla.utils.data_source import DataSource
from nnabla.utils.image_utils import imread, imresize
from nnabla.utils.data_iterator import data_iterator

from neu.datasets import _get_sliced_data_source


def resize_ccrop(img, size, channel_first=True):
    assert isinstance(size, int)
    h1, w1 = img.shape[-2:] if channel_first else img.shape[:-2]
    s = size / min(h1, w1)

    rsz = imresize(img,
                   (max(size, int(round(s * w1))), max(size, int(round(s * h1)))),
                   channel_first=channel_first)

    h2, w2 = rsz.shape[-2:] if channel_first else rsz.shape[:-2]
    h_off = (h2 - size) // 2
    w_off = (w2 - size) // 2
    rsz = rsz[:, h_off:h_off+size, w_off:w_off +
              size] if channel_first else rsz[h_off:h_off+size, w_off:w_off+size]

    h3, w3 = rsz.shape[-2:] if channel_first else rsz.shape[:-2]
    assert h3 == size and w3 == size

    return rsz


class SimpleDatasource(DataSource):
    def __init__(self, img_paths, img_size, labels=None, rng=None, on_memory=True, fix_aspect_ratio=True):
        super(SimpleDatasource, self).__init__(shuffle=True, rng=rng)

        if labels is not None:
            assert len(labels) == len(
                img_paths), "Number of images and labels must be the same."

        self.img_paths = img_paths
        self.labels = labels

        # check resolution
        assert len(img_size) in [1, 2]
        if len(img_size) == 1:
            # If image_size has only one element, deplicate it for both resolution H and W.
            img_size = img_size * 2

        self.im_size = img_size
        self._variables = ["image", "label"]
        self._size = len(self.img_paths)
        self.on_memory = on_memory
        self.fix_aspect_ratio = fix_aspect_ratio

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
        label = 0 if self.labels is None else self.labels[i]

        # keep data paths
        if self.data_history.full():
            self.data_history.get()
        self.data_history.put(self.img_paths[image_idx])

        if self.on_memory and self.images[image_idx] is not None:
            return (self.images[image_idx], label)

        if self.fix_aspect_ratio:
            # perform resize and center crop to keep original aspect ratio.
            img = imread(self.img_paths[image_idx],
                         channel_first=True, num_channels=3)
            img = resize_ccrop(img, self.im_size[0], channel_first=True)
        else:
            # Breaking original aspect ratio, forcely resize image to self.im_size.
            img = imread(
                self.img_paths[image_idx], channel_first=True, size=self.im_size, num_channels=3)

        if self.on_memory:
            self.images[image_idx] = img

        return (img, label)


SUPPORT_IMG_EXTS = [".jpg", ".png"]


def SimpleDataIterator(batch_size, root_dir, image_size,
                       comm=None, shuffle=True, rng=None, on_memory=True, fix_aspect_ratio=True):
    # get all files
    paths = [os.path.join(root_dir, x)
             for x in os.listdir(root_dir) if os.path.splitext(x)[-1] in SUPPORT_IMG_EXTS]

    if len(paths) == 0:
        raise ValueError(f"[SimpleDataIterator] '{root_dir}' is not found. "
                         "Please make sure that you specify the correct directory path.")

    ds = SimpleDatasource(img_paths=paths,
                          img_size=image_size,
                          rng=rng,
                          on_memory=on_memory,
                          fix_aspect_ratio=fix_aspect_ratio)

    logger.info(f"Initialized data iterator. {ds.size} images are found.")

    ds = _get_sliced_data_source(ds, comm, shuffle)

    return data_iterator(ds, batch_size,
                         with_memory_cache=False,
                         use_thread=True,
                         with_file_cache=False)
