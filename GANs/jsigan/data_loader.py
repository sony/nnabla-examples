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
import h5py
import numpy as np
import nnabla as nn


def read_mat_file(data_fname, label_fname, data_name, label_name, train):
    """
    Read training data from .mat file
    """
    data_file = h5py.File(data_fname, 'r')
    label_file = h5py.File(label_fname, 'r')
    data = data_file[data_name][()]
    label = label_file[label_name][()]

    # change type & reorder
    data = np.array(data, dtype=np.float32) / 255.
    label = np.array(label, dtype=np.float32) / 1023.
    data = np.swapaxes(data, 1, 3)
    label = np.swapaxes(label, 1, 3)
    if not train:
        data = nn.Variable.from_numpy_array(data)
        label = nn.Variable.from_numpy_array(label)
    print('.mat files read successfully.')
    return data, label


class JSIData(DataSource):
    def _get_data(self, position):
        if self.train:
            data_train = self.data_train[self._indexes[position]]
            label_train = self.label_train[self._indexes[position]]

            return data_train, label_train

    def __init__(self, conf, train, shuffle=True, rng=None):
        super(JSIData, self).__init__(shuffle=shuffle, rng=rng)
        self.train = train
        self.data_fname = conf.data.lr_sdr_train
        self.label_fname = conf.data.hr_hdr_train
        self.data_name = conf.data.d_name_train
        self.label_name = conf.data.l_name_train
        self.val_data_size = conf.data.val_data_size
        self.data, self.label = read_mat_file(self.data_fname, self.label_fname, self.data_name,
                                              self.label_name, train)
        self.data_train = self.data[:-self.val_data_size, :, :, :]
        self.label_train = self.label[:-self.val_data_size, :, :, :]
        self._size = len(self.data_train)
        self.rng = rng
        self._variables = ('data_train', 'label_train')

    def reset(self):
        if self._shuffle:
            self._indexes = self._rng.permutation(self._size)
        else:
            self._indexes = np.arange(self._size)
        super(JSIData, self).reset()


def jsi_iterator(batch_size, conf, train, with_memory_cache=False, with_file_cache=False, rng=None):
    return data_iterator(JSIData(conf, train, shuffle=True, rng=None), batch_size, rng,
                         with_memory_cache, with_file_cache)
