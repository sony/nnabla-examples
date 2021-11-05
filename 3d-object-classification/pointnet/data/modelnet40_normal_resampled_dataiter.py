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
from typing import Dict, List, Optional, Tuple, Iterable
import numpy as np
from tqdm import tqdm

from nnabla.utils.data_iterator import DataIterator, data_iterator
from nnabla.utils.data_source import DataSource
from nnabla.logger import logger

from .data_utils import (
    load_from_pickle,
    normalize_point_cloud,
    save_as_pickle,
    load_txt_file,
)


def load_dataset_path_file(data_dir: str, txt_file_name: str) -> List[str]:
    data_lists = load_txt_file(os.path.join(data_dir, txt_file_name))
    shape_names = ["_".join(x.split("_")[0:-1]) for x in data_lists]
    data_paths = [
        (shape_names[i], os.path.join(data_dir, shape_names[i], data_lists[i]) + ".txt") for i in range(len(data_lists))
    ]
    return data_paths


def load_txt_as_np_array(
    data_paths: Iterable[Tuple[str, str]], num_points: int, classes_dict: Dict[str, int]
) -> Tuple[np.ndarray, np.ndarray]:
    point_cloud_data = []
    label_data = []

    for data_path in tqdm(data_paths, total=len(data_paths)):
        class_name, point_cloud_data_path = data_path
        # point cloud
        point_cloud = np.loadtxt(
            point_cloud_data_path, delimiter=",").astype(np.float32)
        point_cloud = point_cloud[:num_points, :]
        point_cloud_data.append(point_cloud)
        # label
        label = int(classes_dict[class_name])
        label_data.append(label)

    return np.array(point_cloud_data, dtype=np.float32), np.array(label_data, dtype=np.int32)


class ModelNet40NormalResampledDataset(DataSource):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        train: bool,
        shuffle: bool,
        num_points: int,
        normalize: bool,
        rng: Optional[int] = None,
    ) -> None:
        super().__init__(shuffle=shuffle, rng=rng)
        self._shuffle = shuffle
        self._train = train
        self._batch_size = batch_size
        self._normalize = normalize

        if self._train:
            processed_data_path = os.path.join(
                data_dir, "train_modelnet40_normal_resampled.pkl")
        else:
            processed_data_path = os.path.join(
                data_dir, "test_modelnet40_normal_resampled.pkl")

        self._shape_names = load_txt_file(
            os.path.join(data_dir, "modelnet40_shape_names.txt"))
        self._classes_dict = dict(
            zip(self._shape_names, range(len(self._shape_names))))

        if not os.path.exists(processed_data_path):
            logger.info("Load from original datasets ...")
            txt_file_name = "modelnet40_train.txt" if train else "modelnet40_test.txt"
            data_paths = load_dataset_path_file(data_dir, txt_file_name)
            self._point_clouds, self._labels = load_txt_as_np_array(
                data_paths, num_points, self._classes_dict)
            logger.info(f"Saving data as pkl ... to {processed_data_path}")
            save_as_pickle((self._point_clouds, self._labels),
                           processed_data_path)
        else:
            logger.info("Load from processed datasets ...")
            processed_data = load_from_pickle(processed_data_path)
            self._point_clouds, self._labels = processed_data

        self._size = len(self._point_clouds)
        self._variables = ("point_cloud", "label")
        if rng is None:
            rng = np.random.RandomState(313)
        self.rng = rng
        self.reset()

    def _get_data(self, position):
        index = self._indexes[position]
        point_cloud = np.array(self._point_clouds[index], dtype=np.float32)[
                               np.newaxis, :, :3]  # not use normal vector
        label = np.array([self._labels[index]], dtype=np.int32)

        if self._normalize:
            point_cloud = normalize_point_cloud(point_cloud)

        return (np.squeeze(point_cloud), label)

    def reset(self) -> None:
        if self._shuffle:
            self._indexes = self.rng.permutation(self._size)
        else:
            self._indexes = np.arange(self._size)
        super(ModelNet40NormalResampledDataset, self).reset()


def data_iterator_modelnet40_normal_resampled(
    data_dir: str,
    batch_size: int,
    train: bool,
    shuffle: bool,
    num_points: int,
    normalize: bool,
    stop_exhausted: bool = True,
    with_memory_cache: bool = True,
    with_file_cache: bool = False,
    rng: Optional[int] = None,
) -> DataIterator:
    dataset = ModelNet40NormalResampledDataset(
        data_dir,
        batch_size,
        train,
        shuffle,
        num_points,
        normalize,
    )
    return data_iterator(
        dataset,
        batch_size,
        rng=rng,
        with_memory_cache=with_memory_cache,
        with_file_cache=with_file_cache,
        stop_exhausted=stop_exhausted,
    )
