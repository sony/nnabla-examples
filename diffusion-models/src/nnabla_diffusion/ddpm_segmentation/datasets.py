# Copyright 2023 Sony Group Corporation.
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

import blobfile as bf
import nnabla
import numpy as np
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource
from nnabla_diffusion.config.python.dataset import DatasetConfig
from nnabla_diffusion.dataset.common import SimpleDatasource
from nnabla_diffusion.ddpm_segmentation.data_util import get_palette
from nnabla_diffusion.ddpm_segmentation.utils import to_labels
from PIL import Image

SUPPORT_IMG_EXTS = [".jpg", ".png", ".jpeg"]


def ImageLabelDataIterator(conf: DatasetConfig, num_images, comm, label_creator_callback=None, rng=None):

    paths = [os.path.join(conf.dataset_root_dir, x)
             for x in sorted(os.listdir(conf.dataset_root_dir)) if os.path.splitext(x)[-1] in SUPPORT_IMG_EXTS][:num_images]

    labels = None
    if label_creator_callback is not None:
        labels = [np.load(label_creator_callback(path)) for path in paths]

    ds = SimpleDatasource(
        conf,
        img_paths=paths,
        labels=labels,
        rng=rng
    )

    return data_iterator(ds,
                         1,
                         with_memory_cache=False,
                         use_thread=True,
                         with_file_cache=False)


class PixelWiseDataSource(DataSource):
    def __init__(self, conf: DatasetConfig, features, labels, rng=None):
        super(PixelWiseDataSource, self).__init__(shuffle=conf.shuffle_dataset,
                                                  rng=rng)

        self.features = features
        self.labels = labels
        self._variables = ["feature", "label"]
        self._size = len(self.features)

        self.reset()

    def __len__(self):
        return self._size

    def reset(self):
        self._indexes = self._rng.permutation(
            self.size) if self.shuffle else np.arange(self.size)
        super(PixelWiseDataSource, self).reset()

    def _get_data(self, i):
        feature_idx = self._indexes[i]

        return (self.features[feature_idx], self.labels[feature_idx])


def PixelWiseDataIterator(conf: DatasetConfig, features, labels):
    return data_iterator(PixelWiseDataSource(conf, features, labels),
                         batch_size=conf.batch_size,
                         with_memory_cache=False,
                         use_thread=True,
                         with_file_cache=False
                         )


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = os.path.splitext(entry)[-1]

        if bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
            continue
        elif ext.lower() in ["jpg", "jpeg", "png"]:
            results.append(full_path)

    return results
