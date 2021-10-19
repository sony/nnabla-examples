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
import queue
import sys
import tarfile

import numpy as np
from nnabla.logger import logger
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource
from nnabla.utils.download import download

from neu.datasets import _get_sliced_data_source


class Cifar10DataSource(DataSource):
    '''
    Get data directly from cifar10 dataset from Internet(yann.lecun.com).
    '''

    def _get_data(self, position):
        image = self._images[self._indexes[position]]
        label = self._labels[self._indexes[position]]

        # keep data paths
        if self.data_history.full():
            self.data_history.get()
        self.data_history.put(self._indexes[position])

        return (image, label)

    def __init__(self, train=True, shuffle=False, rng=None):
        super(Cifar10DataSource, self).__init__(shuffle=shuffle, rng=rng)

        self._train = train
        data_uri = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        logger.info('Getting labeled data from {}.'.format(data_uri))
        r = download(data_uri)  # file object returned
        with tarfile.open(fileobj=r, mode="r:gz") as fpin:
            # Training data
            if train:
                images = []
                labels = []
                for member in fpin.getmembers():
                    if "data_batch" not in member.name:
                        continue
                    fp = fpin.extractfile(member)
                    data = np.load(fp, encoding="bytes", allow_pickle=True)
                    images.append(data[b"data"])
                    labels.append(data[b"labels"])
                self._size = 50000
                self._images = np.concatenate(
                    images).reshape(self._size, 3, 32, 32)
                self._labels = np.concatenate(labels).reshape(-1, 1)
            # Validation data
            else:
                for member in fpin.getmembers():
                    if "test_batch" not in member.name:
                        continue
                    fp = fpin.extractfile(member)
                    data = np.load(fp, encoding="bytes", allow_pickle=True)
                    images = data[b"data"]
                    labels = data[b"labels"]
                self._size = 10000
                self._images = images.reshape(self._size, 3, 32, 32)
                self._labels = np.array(labels).reshape(-1, 1)
        r.close()
        logger.info('Getting labeled data from {}.'.format(data_uri))

        self._size = self._labels.size
        self._variables = ('x', 'y')
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng

        # keep data paths
        self.data_history = queue.Queue(maxsize=128)

        self.reset()

    def get_last_data_path(self, size=1):
        ret = ()
        for i in range(size):
            ret += (self.data_history.get(), )

        return ret

    def reset(self):
        if self._shuffle:
            self._indexes = self.rng.permutation(self._size)
        else:
            self._indexes = np.arange(self._size)
        super(Cifar10DataSource, self).reset()

    @property
    def images(self):
        """Get copy of whole data with a shape of (N, 1, H, W)."""
        return self._images.copy()

    @property
    def labels(self):
        """Get copy of whole label with a shape of (N, 1)."""
        return self._labels.copy()


def Cifar10DataIterator(batch_size, image_size=(32, 32), comm=None, shuffle=True, rng=None, train=True):
    ds = Cifar10DataSource(train=train, shuffle=shuffle, rng=rng)

    ds = _get_sliced_data_source(ds, comm, shuffle)

    return data_iterator(ds, batch_size,
                         with_memory_cache=False,
                         use_thread=True,
                         with_file_cache=False)
