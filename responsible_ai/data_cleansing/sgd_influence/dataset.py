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

import os
import numpy as np
import nnabla as nn
from nnabla.utils.data_source_loader import _load_functions
from nnabla.utils.data_source_implements import CsvDataSource
from nnabla.utils.image_utils import imresize
import nnabla.functions as F
import random


class MeanStd(object):
    def __new__(cls, *args, **kargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super(MeanStd, cls).__new__(cls)
        return cls._instance

    def __init__(self, resize_size, n_channels):
        if n_channels == 1:
            self._init_single_channel(resize_size)
        else:
            self._init_multi_channel(resize_size)

    def _init_single_channel(self, resize_size):
        ch_mean = np.array([np.full(resize_size, 0.5)])
        ch_std = np.array([np.full(resize_size, 0.5)])
        self.ch_mean = nn.Variable.from_numpy_array(ch_mean)
        self.ch_std = nn.Variable.from_numpy_array(ch_std)

    def _init_multi_channel(self, resize_size):
        ch_mean = np.array([np.full(resize_size, 0.4914), np.full(
            resize_size, 0.4822), np.full(resize_size, 0.4465)])
        ch_std = np.array([np.full(resize_size, 0.2023), np.full(
            resize_size, 0.1994), np.full(resize_size, 0.2010)])
        self.ch_mean = nn.Variable.from_numpy_array(ch_mean)
        self.ch_std = nn.Variable.from_numpy_array(ch_std)


class CsvDataSourceFilename(CsvDataSource):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._filepath_column_idx = self._get_column_idx('filepath')
        self._label_column_idx = self._get_column_idx('label')

    def _get_column_idx(self, target):
        """
        Parameters
        ----------
        target: str
            'filepath' or 'label'

        Returns
        ----------
        i: int
            index of column that contains 'filepath' or 'label' info
        """
        if target not in ['filepath', 'label']:
            raise KeyError("target is 'filepath' or 'label'")

        def _is_filepath_idx(value):
            ext = (os.path.splitext(value)[1]).lower()
            if ext in _load_functions.keys():
                return True
            return False

        def _is_label_idx(value):
            try:
                value = float(value)
                return True
            except ValueError:
                return False

        func_dict = {
            'filepath': _is_filepath_idx,
            'label': _is_label_idx
        }

        # judge from first row
        idx = 0
        for i, column_value in enumerate(self._rows[idx]):
            # Implemented refering to below
            # https://github.com/sony/nnabla/blob/f5eff2de5329ef02c40e7a5d7344abd91b19ece8/python/src/nnabla/utils/data_source_implements.py#L402
            # https://github.com/sony/nnabla/blob/f5eff2de5329ef02c40e7a5d7344abd91b19ece8/python/src/nnabla/utils/data_source_loader.py#L343
            if func_dict[target](column_value):
                return i
        raise RuntimeError(
            '{} info is not in {}.'.format(target, self._filename))

    def get_filepath_to_data(self, idx):
        """
        Parameters
        ----------
        idx: int
            index of self._rows that includes filepath to data

        Returns
        ----------
        filepath: str
            ex: 'training/1.png'
        """
        return self._rows[idx][self._filepath_column_idx]

    def get_n_classes(self):
        """
        Parameters
        ----------
        None

        Returns
        ----------
        n_classes: int
            number of classes of label
            ex. n_classes is 10 if the dataset contains 0-9 labels 
        """

        labels = [row[self._label_column_idx] for row in self._rows]
        n_classes = len(set(labels))
        return n_classes


def setup_nnabla_dataset(filename):
    dataset = CsvDataSourceFilename(
        filename=filename,
        shuffle=False,
        normalize=True
    )
    return dataset


def get_batch_data(dataset, idx_list_to_data, idx_list_to_idx, resize_size, test, seeds=None, escape_list=[]):
    X = []
    y = []
    if seeds is None:
        seeds = [0] * len(idx_list_to_idx)
    for idx, seed in zip(idx_list_to_idx, seeds):
        if idx_list_to_data[idx] in escape_list:
            continue
        image, label = get_data(
            dataset, idx_list_to_data[idx], resize_size, test, seed)
        X.append(image)
        y.append(label)
    y = np.array(y).reshape(-1, 1)
    return X, y


def normalize(x, resize_size):
    mean_std = MeanStd(resize_size, x.shape[0])
    ch_mean = mean_std.ch_mean
    ch_std = mean_std.ch_std
    x = F.sub2(x, ch_mean)
    x = F.div2(x, ch_std)
    return x


def transform(x, resize_size, seed, test):
    x = nn.Variable.from_numpy_array(x)
    if not test:
        # random crop
        random.seed(seed)
        x = F.image_augmentation(x, shape=resize_size, pad=(4, 4), seed=seed)
        # random horizontal flip
        random.seed(seed)
        x = F.image_augmentation(x, flip_lr=True, seed=seed)
    x = normalize(x, resize_size)
    x.forward(clear_buffer=True)
    x = x.d
    return x


def init_dataset(input_train, input_val, seed):
    return init_nnabla(input_train, input_val)


def init_nnabla(input_train, input_val):
    trainset = setup_nnabla_dataset(input_train)
    valset = setup_nnabla_dataset(input_val)
    image, _ = trainset._get_data(0)
    n_classes = trainset.get_n_classes()
    ntr = trainset.size
    nval = valset.size
    return trainset, valset, image.shape, n_classes, ntr, nval


def get_data(dataset, idx, resize_size, test, seed):
    image, label = get_data_nnabla(dataset, idx, resize_size, test, seed)
    return image, label


def get_data_nnabla(dataset, idx, resize_size, test, seed):
    image, label = dataset._get_data(idx)
    image = imresize(image, resize_size, channel_first=True)
    image = transform(image, resize_size, seed, test)
    return image, label[0]


def get_batch_indices(num_data, batch_size, seed=None):
    if seed is None:
        shuffled_idx = np.arange(num_data)
    else:
        np.random.seed(seed)
        shuffled_idx = np.random.permutation(num_data)
    indices_list = []
    for i in range(0, num_data, batch_size):
        indices = shuffled_idx[i:i+batch_size]
        indices_list.append(indices)
    return indices_list


def get_image_size(image_shape, max_size=128, min_size=32):
    """
    Parameters
    ----------
    image_shape: tuple
        (int, int)
    max_size: int
    min_size: int

    Returns
    ----------
    resize_size: tuple
        (int, int)
    """
    h, w = image_shape
    short_edge = min(h, w)
    if short_edge < min_size:
        resize_size = (min_size, min_size)
    elif (short_edge >= min_size) & (short_edge <= max_size):
        resize_size = (short_edge, short_edge)
    elif short_edge > max_size:
        resize_size = (max_size, max_size)
    return resize_size
