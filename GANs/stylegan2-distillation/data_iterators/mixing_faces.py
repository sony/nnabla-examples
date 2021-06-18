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

import os
from PIL import Image
import numpy as np


class MixingFacesData(DataSource):

    def __init__(self, data_root, image_size, img_exts, rng=313):
        super(MixingFacesData, self).__init__(shuffle=True, rng=rng)

        self.img_exts = img_exts

        self.paths_c = sorted([os.path.join(data_root, img_path)
                               for img_path in os.listdir(data_root) if img_exts[0] in img_path])
        self.paths_s = sorted([os.path.join(data_root, img_path)
                               for img_path in os.listdir(data_root) if img_exts[1] in img_path])
        self.paths_m = sorted([os.path.join(data_root, img_path)
                               for img_path in os.listdir(data_root) if img_exts[2] in img_path])

        assert len(self.paths_c) > 0 and len(self.paths_s) > 0 and len(
            self.paths_m), 'No images were found in {}'.format(data_root)
        assert len(self.paths_c) == len(self.paths_s) == len(self.paths_m), 'Unequal images of two domains found {} != {} != {}'.format(
            len(self.paths_c), len(self.paths_s), len(self.paths_m))

        self.image_size = image_size

        self._variables = ['image_cs', 'image_m']

        self._size = self.__len__()

    def __len__(self):
        return len(self.paths_m)

    @property
    def size(self):
        return self._size

    def reset(self):
        if self._shuffle:
            self._indexes = np.arange(self.size)
            np.random.shuffle(self._indexes)
        else:
            self._indexes = np.arange(self._size)
        super(MixingFacesData, self).reset()

    def get_image(self, path):
        image = Image.open(path).convert("RGB").resize(
            self.image_size, resample=Image.BILINEAR)
        image = np.array(image)/255.0
        image = np.transpose(image.astype(np.float32), (2, 0, 1))

        image = (image - 0.5)/(0.5)

        return image

    def _get_data(self, i):
        image_path_c = self.paths_c[i]
        image_path_s = image_path_c.replace(self.img_exts[0], self.img_exts[1])
        image_path_m = image_path_c.replace(self.img_exts[0], self.img_exts[2])
        image_c = self.get_image(image_path_c)
        image_s = self.get_image(image_path_s)
        image_m = self.get_image(image_path_m)

        image_cs = np.concatenate((image_c, image_s), axis=0)

        return (image_m, image_cs)


def get_data_iterator_mix(data_root, comm, batch_size, image_size, img_exts=['content.png', 'style.png', 'mix.png']):

    data_source = MixingFacesData(data_root, image_size, img_exts)

    data_iterator_ = data_iterator(data_source, batch_size=batch_size)

    if comm is not None:
        if comm.n_procs > 1:
            data_iterator_ = data_iterator_.slice(
                rng=None, num_of_slices=comm.n_procs, slice_pos=comm.rank)

    return data_iterator_
