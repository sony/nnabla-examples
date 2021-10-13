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

from nnabla.utils.data_source import DataSource
from nnabla.utils.data_iterator import data_iterator
import nnabla as nn
import nnabla.functions as F

import os
from pathlib import Path
from PIL import Image
import numpy as np

EXTS = ['jpg', 'jpeg', 'png']


class FFHQData(DataSource):

    def _get_data(self, i):
        img_path = self.paths[i]
        image = Image.open(img_path).convert("RGB").resize(
            (self.img_size, self.img_size), resample=Image.BILINEAR)
        image = np.array(image)/255.0
        image = np.transpose(image.astype(np.float32), (2, 0, 1))

        image[0] = (image[0] - 0.5)/(0.5)
        image[1] = (image[1] - 0.5)/(0.5)
        image[2] = (image[2] - 0.5)/(0.5)

        return (image,)

    def __init__(self, data_config, img_size):
        super(FFHQData, self).__init__(
            shuffle=data_config['shuffle'], rng=data_config['rng'])

        self.paths = [p for ext in EXTS for p in Path(
            data_config["path"]).glob('**/*.{}'.format(ext))]

        assert len(self.paths) > 0, 'No images were found in {}'.format(
            data_config["path"])

        self.img_size = img_size

        self._variables = ['image']

        self._size = self.__len__()

    def __len__(self):
        return len(self.paths)

    @property
    def size(self):
        return self._size

    def reset(self):
        # reset method initialize self._indexes
        if self._shuffle:
            self._indexes = np.arange(self.size)
            np.random.shuffle(self._indexes)
        else:
            self._indexes = np.arange(self._size)
        super(FFHQData, self).reset()


def get_data_iterator_ffhq(data_config, batch_size, img_size, comm):

    data_source = FFHQData(data_config, img_size)
    data_iterator_ffhq = data_iterator(data_source, batch_size=batch_size)

    if comm is not None:
        if comm.n_procs > 1:
            data_iterator_ffhq = data_iterator_ffhq.slice(
                rng=None, num_of_slices=comm.n_procs, slice_pos=comm.rank)

    return data_iterator_ffhq
