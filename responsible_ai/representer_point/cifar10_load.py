# Copyright (c) 2021 Sony Group Corporation. All Rights Reserved.
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
"""
Provide data iterator for CIFAR10 examples.
"""
import numpy as np
import tarfile
import random
from nnabla.utils.data_iterator import data_iterator
from nnabla.logger import logger
from nnabla.utils.data_source import DataSource
from nnabla.utils.data_source_loader import download


class Cifar10NumpySource(DataSource):
    def __init__(self, x, y, y_shuffled=False, shuffle=False, rng=None):
        super(Cifar10NumpySource, self).__init__(shuffle=shuffle, rng=rng)
        self._images = x
        self._labels = y
        if y_shuffled is not False:
            self._shuffle_labels = y_shuffled
        else:
            self._shuffle_labels = False
        self._shuffle = shuffle
        self._size = self._labels.size
        if self._shuffle_labels is not False:
            self._variables = ("x", "y", "shuffle")
        else:
            self._variables = ("x", "y")
        if rng is None:
            rng = np.random.RandomState(313)
        self.rng = rng
        self.reset()

    def _get_data(self, position):
        image = self._images[self._indexes[position]]
        label = self._labels[self._indexes[position]]

        if self._shuffle_labels is not False:
            shuffle = self._shuffle_labels[self._indexes[position]]
            return (image, label, shuffle)
        else:

            return (image, label)

    def reset(self):
        if self._shuffle:
            self._indexes = self.rng.permutation(self._size)
        else:
            self._indexes = np.arange(self._size)
        super(Cifar10NumpySource, self).reset()

    @property
    def images(self):
        """Get copy of whole data with a shape of (N, 1, H, W)."""
        return self._images.copy()

    @property
    def labels(self):
        """Get copy of whole label with a shape of (N, 1)."""
        return self._labels.copy()

    @property
    def shuffled_labels(self):
        """Get copy of whole label with a shape of (N, 1)."""
        if self._shuffle_labels:
            return self._shuffle_labels.copy()


class Cifar10DataSource(DataSource):
    """
    Get data directly from cifar10 dataset from Internet(yann.lecun.com).
    """

    def __init__(
        self,
        train=True,
        shuffle=False,
        rng=None,
        label_shuffle=False,
        label_shuffle_rate=0.1,
    ):
        print("label_shuffled: ", label_shuffle)
        super(Cifar10DataSource, self).__init__(shuffle=shuffle, rng=rng)

        self._train = train
        self._label_shuffle = label_shuffle
        self._shuffle = shuffle

        data_uri = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        logger.info("Getting labeled data from {}.".format(data_uri))
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
                self.raw_label = self._labels.copy()

                if label_shuffle:
                    self.shuffle_rate = label_shuffle_rate
                    self.label_shuffle()
                    print(f"{self.shuffle_rate*100}% of data was shuffled ")
                    print(len(np.where(self._labels != self.raw_label)[0]))
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
                self.raw_label = self._labels.copy()
        r.close()
        logger.info("Getting labeled data from {}.".format(data_uri))

        self._size = self._labels.size
        if self._label_shuffle is not False:
            self._variables = ("x", "y", "shuffle")
        else:
            self._variables = ("x", "y", "y")
        if rng is None:
            rng = np.random.RandomState(313)
        self.rng = rng
        self.reset()

    def _get_data(self, position):
        image = self._images[self._indexes[position]]
        label = self.raw_label[self._indexes[position]]

        if self._label_shuffle:
            shuffle = self._labels[self._indexes[position]]
            return (image, label, shuffle)
        else:
            return (image, label, label)

    def label_shuffle(self):
        num_cls = int(np.max(self._labels)) + 1
        raw_label = self._labels.copy()
        extract_num = int(len(self._labels) * self.shuffle_rate // 10)
        for i in range(num_cls):
            extract_ind = np.where(raw_label == i)[0]
            labels = [j for j in range(num_cls)]
            labels.remove(i)  # candidate of shuffle label
            artificial_label = [
                labels[int(i) % (num_cls - 1)] for i in range(int(extract_num))
            ]
            artificial_label = np.array(
                random.sample(artificial_label, len(artificial_label))
            ).astype("float32")
            convert_label = np.array([i for _ in range(len(extract_ind))])
            convert_label[-extract_num:] = artificial_label

            self._labels[extract_ind] = convert_label.reshape(-1, 1)

    def reset(self):
        if self._shuffle:
            self._indexes = self.rng.permutation(self._size)
        else:
            self._indexes = np.arange(self._size)
        super(Cifar10DataSource, self).reset()

    @property
    def _raw_label(self):
        return self.raw_label.copy()

    @property
    def images(self):
        """Get copy of whole data with a shape of (N, 1, H, W)."""
        return self._images.copy()

    @property
    def labels(self):
        """Get copy of whole label with a shape of (N, 1)."""
        return self._labels.copy()


def data_source_cifar10(train=True, rng=None, shuffle=True, label_shuffle=False):
    """
    Provide DataIterator with :py:class:`Cifar10DataSource`
    with_memory_cache and with_file_cache option's default value is all False,
    because :py:class:`Cifar10DataSource` is able to store all data into memory.
    """
    return Cifar10DataSource(
        train=train, shuffle=shuffle, rng=rng, label_shuffle=label_shuffle
    )


def data_iterator_cifar10(
    batch_size,
    train=True,
    rng=None,
    shuffle=True,
    with_memory_cache=False,
    with_file_cache=False,
):
    """
    Provide DataIterator with :py:class:`Cifar10DataSource`
    with_memory_cache and with_file_cache option's default value is all False,
    because :py:class:`Cifar10DataSource` is able to store all data into memory.

    """
    return data_iterator(
        Cifar10DataSource(train=train, shuffle=shuffle, rng=rng),
        batch_size,
        rng,
        with_memory_cache,
        with_file_cache,
    )
